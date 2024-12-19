import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from tqdm import tqdm
import json
from torchvision.utils import save_image

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
                #_, _, text_feature = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()
                #breakpoint()
                #logits = model.logit_head(image_features) 
                #breakpoint()
                #logits = image_features @ image_features.t()  #takes on img feat
                loss = loss_fn(score, feat, target, target_cam, logits)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_ttadapt(cfg,
                 model,
                 prompt_learner,
                 val_loader,
                 num_query, 
                 optimizer,
                 scheduler,
                 local_rank, segmentor=None, path=None, save_sample=False,):
    
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.tta")
    logger.info('start test time adapt')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    with torch.no_grad():
        
        for n_iter, (img, seg_img, vid, target_cam, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Collecting Query Image features!"):
            #breakpoint()
            # if n_iter == num_query: break
            img = img.to(device)
            target = vid
            with amp.autocast(enabled=True):
                image_feature = model(img, get_image = True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        #breakpoint()
        labels_list = torch.tensor(labels[:num_query]).cuda() #torch.stack(torch.tensor(labels), dim=0).cuda() #N
        image_features_list = torch.stack(image_features[:num_query], dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features
    #breakpoint()
    model = model.eval()
    text_feature_dict = {}
    for epoch in range(1, epochs + 1):
        
        if epoch > 1 and epoch % 2 == 0:
            print(f"Evaluating on epoch {epoch}")
            model.eval()
            for n_iter, (img, seg_img, vid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating!"):
                with torch.no_grad():
                    img = img.to(device)
                    image_feature = model(img, get_image=True)
                    evaluator.update((image_feature, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute(textfeats=text_feature_dict)
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

        loss_meter.reset()
        scheduler.step(epoch)
        iter_list = torch.randperm(num_image).to(device)
        for i in tqdm(range(i_ter+1), total=i_ter+1, desc="Iterating epoch values!"):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                prompts = prompt_learner(target) 
                text_features = model.text_encoder(prompts, prompt_learner.tokenized_prompts)
                #breakpoint()
                
                #text_features = model(label = target, get_text = True)
            #if i == i_ter: breakpoint()
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)
            
            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            for j in range(text_features.shape[0]):
                text_feature_dict[labels_list[b_list[j]].item()] = text_features[j].detach().cpu()

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(val_loader),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            torch.save(prompt_learner.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ttaPL_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("TTA running time: {}".format(total_time))


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, segmentor=None, path=None, save_sample=False):
    # breakpoint()
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
        if segmentor is not None: segmentor.to(device)

    model.eval()
    
    img_path_list = []

    for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            #breakpoint()
            img = img.to(device)
            if segmentor is not None:
                #breakpoint()
                seg_img.to(device)
                prompt = ["the person in the image"] * seg_img.shape[0]
                mask = segmentor(seg_img, prompt)[0]
                mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
                mask = torch.sigmoid(mask)
                #breakpoint()
                threshold = 0.5
                mask[mask >= threshold] = 1.0
                mask[mask < threshold] = 0.0
                img = img #* (1.0 - mask)
                
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            #breakpoint()
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
            if save_sample:
                img_new = img.detach().cpu()
                img1 = img_new[0]
                save_image(img1, os.path.join(path, "sample.png"))


    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    if path is not None:
        torch.save(distmat, f"{path}/distmat_og.pth", pickle_protocol=4)
        torch.save(qf, f"{path}/qf_og.pth", pickle_protocol=4)
        torch.save(gf, f"{path}/gf_og.pth", pickle_protocol=4)
        with open(f"{path}/imgpaths.json", 'w') as f:
            f.write(json.dumps(img_path_list))
        torch.save(pids, f"{path}/pids.pth", pickle_protocol=4)
        torch.save(camids, f"{path}/camids.pth", pickle_protocol=4)


    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


# def do_inference_camidwise(cfg,
#                  model,
#                  val_loader,
#                  num_query):
#     # breakpoint()
#     device = "cuda"
#     logger = logging.getLogger("transreid.test")
#     logger.info("Enter inferencing")

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

#     evaluator.reset()

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device)

#     model.eval()
#     img_path_list = []
#     feats = {}
#     for i in range(15):
#         feats[i] = []
#     for n_iter, (img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
#         with torch.no_grad():
#             #breakpoint()
#             img = img.to(device)
#             if cfg.MODEL.SIE_CAMERA:
#                 camids = camids.to(device)
#             else: 
#                 camids = None
#             if cfg.MODEL.SIE_VIEW:
#                 target_view = target_view.to(device)
#             else: 
#                 target_view = None
#             feat = model(img, cam_label=camids, view_label=target_view)
#             feats[camid[0]].append((feat.detach().cpu(), pid, camid))

#             #evaluator.update((feat, pid, camid))
#             img_path_list.extend(imgpath)

#     for key, feat in feats.items():
#         evaluator.reset()
#         for f in feat:
#             evaluator.update((f[0].to(device), f[1], f[2]))
#         print("Evaluating for cam id", key)
#         cmc, mAP, _, _, _, _, _ = evaluator.compute()
#         logger.info("Validation Results ")
#         #logger.info("mAP: {:.1%}".format(mAP))
#         logger.info("mAP: {}".format(mAP))
#         for r in [1, 5, 10]:
#             #logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[:][r - 1]))
#             logger.info("CMC curve, Rank-{}:{}".format(r, cmc[:][r - 1]))
    

def save_features(cfg,
                 model,
                 val_loader,
                 num_query,
                 filepath):
    # breakpoint()
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    feats = {}
    for i in range(15):
        feats[i] = []
    
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            feats[camid[0]].append(feat[0].detach().cpu().numpy().tolist())
            
            img_path_list.extend(imgpath)
    
    for k,v in feats.items():
        print(k, len(v))

    with open(os.path.join(filepath, "features_10cams.json"), 'w') as f:
        f.write(json.dumps(feats))