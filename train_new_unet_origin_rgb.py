if __name__ == '__main__':
    import sys
    sys.path.append('./model')
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from model_new_unet_origin_rgb import Generator,Loss
    from option import TrainOption, TestOption
    from dataload_new import MyDataset3
    from utils import Manager, update_lr
    import datetime

    # torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    os.environ['CUDA_VISION_DEVICES'] = '0,1,2,3,4,5,6,7'

    DEVICE = torch.device('cuda:0')
    lr = opt.lr

    # dataset = MyDataset3('../../../../../data1/hongyang/OE_SYNdataset_new','../train_shai1.txt')
    dataset = MyDataset3('../../../../../data4/hongyang/REAL_DATASET_NEW', '../train_real.txt')
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=opt.shuffle,
                             pin_memory=True)

    test_opt = TestOption().parse()
    # test_dataset = MyDataset3('../../../../../data1/hongyang/OE_SYNdataset_new','../test_shai2.txt')
    test_dataset = MyDataset3('../../../../../data4/hongyang/REAL_DATASET_NEW', '../test_real.txt')
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=test_opt.batch_size,
                                  num_workers=test_opt.n_workers,
                                  pin_memory=True)

    G1 = Generator().to(DEVICE)
    G_static = torch.load('./checkpoints/aia2ha_new_unet_origin_all_rgb/Model/height_512/400_G.pt')
    model_static = {k.replace('module.', ''): v for k, v in G_static.items()}

    # G_static = torch.load('./checkpoints/aia2ha_new_unet_new_sg/Model/height_512/320_G.pt')
    # model_static = {k.replace('module.', ''): v for k, v in G_static.items()}
    # # model_static.update(G_static)
    G1.load_state_dict(model_static)

    G=nn.DataParallel(G1,device_ids=[0,1,2,3,4,5,6,7])

    criterion = Loss(opt,DEVICE)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    # G_optim1 = torch.optim.Adam(G1.stage1.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    manager = Manager(opt)

    current_step = 0
    total_step = opt.n_epochs * len(data_loader)
    start_time = datetime.datetime.now()
    for epoch in range(1, opt.n_epochs + 1):
        for _, (input, target) in enumerate(data_loader):
            current_step += 1
            input, target = input.to(DEVICE,non_blocking=True), target.to(DEVICE,non_blocking=True)
            G_loss,target_tensor, generated_tensor = criterion(G, input, target)

            # G_optim1.zero_grad()
            # G_loss1.backward()
            # G_optim1.step()

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()



            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'G_loss': G_loss.detach().item(),
                       'G_state_dict': G.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor.detach()}

            manager(package)

            if opt.debug:
                break

        if epoch % opt.epoch_save == 0:
            torch.save(G.state_dict(), os.path.join(test_opt.model_dir, "{:d}_G.pt".format(epoch)))

            image_dir = os.path.join(test_opt.image_dir, "{:d}".format(epoch))
            os.makedirs(image_dir, exist_ok=True)
            with torch.no_grad():
                for i, (input, target) in enumerate(test_data_loader):
                    input = input.to(DEVICE)
                    fake = G(input)
                    manager.save_image(fake, os.path.join(image_dir, "{:d}_fake.png".format(i)))
                    manager.save_image(target, os.path.join(image_dir, "{:d}_real.png".format(i)))

        if epoch > opt.epoch_decay:
            lr = update_lr(opt.lr, lr, opt.n_epochs - opt.epoch_decay, G_optim)

    print("Total time taken: ", datetime.datetime.now() - start_time)
