if __name__ == '__main__':
    import sys

    sys.path.append('../model')
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from model_new_unet_sg import Generator,Loss
    from model_new_unet_unetatt import Generator as Generator1
    from option import TrainOption, TestOption
    from dataload_new import MyDataset2
    from utils import Manager, update_lr
    import datetime

    # torch.backends.cudnn.benchmark = True

    opt = TrainOption().parse()
    os.environ['CUDA_VISION_DEVICES'] = '2,3,4,6,7'

    DEVICE = torch.device('cuda:2')
    lr1 = opt.lr
    lr2 = opt.lr

    dataset = MyDataset2('../../../../../data1/hongyang/OE_SYNdataset_new','../train_shai1.txt')
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=opt.shuffle,
                             pin_memory=True)

    test_opt = TestOption().parse()
    test_dataset = MyDataset2('../../../../../data1/hongyang/OE_SYNdataset_new','../test_shai1.txt')
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=test_opt.batch_size,
                                  num_workers=test_opt.n_workers,
                                  pin_memory=True)

    G1 = Generator().to(DEVICE)
    G2 = Generator().to(DEVICE)
    # G_static = torch.load('./checkpoints/aia2ha_new_unet_SG/Model/height_512/260_G.pt')
    # model_static = {k.replace('module.', ''): v for k, v in G_static.items()}
    # # model_static.update(G_static)
    # G1.load_state_dict(model_static)

    G1 = nn.DataParallel(G1,device_ids=[2,3,4,6,7])
    G2 = nn.DataParallel(G2, device_ids=[2, 3, 4, 6, 7])

    criterion = Loss(opt,DEVICE)

    G_optim1 = torch.optim.Adam(G1.parameters(), lr=lr1, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    G_optim2 = torch.optim.Adam(G2.parameters(), lr=lr2, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    # G_optim1 = torch.optim.Adam(G1.stage1.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    manager = Manager(opt)

    current_step = 0
    total_step = opt.n_epochs * len(data_loader)
    start_time = datetime.datetime.now()
    for epoch in range(1, opt.n_epochs + 1):
        epoch=epoch
        for _, (input, target) in enumerate(data_loader):
            current_step += 1
            input, target = input.to(DEVICE,non_blocking=True), target.to(DEVICE,non_blocking=True)
            G_loss1,target_tensor1, generated_tensor1 = criterion(G1, input, target)
            G_loss2, target_tensor2, generated_tensor2 = criterion(G2, input, target)
            # G_optim1.zero_grad()
            # G_loss1.backward()
            # G_optim1.step()

            G_optim1.zero_grad()
            G_loss1.backward()
            G_optim1.step()

            G_optim2.zero_grad()
            G_loss2.backward()
            G_optim2.step()



            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'G1_loss': G_loss1.detach().item(),
                       'G1_state_dict': G1.state_dict(),
                       'target_tensor': target_tensor1,
                       'generated_tensor1': generated_tensor1.detach(),
                       'G2_loss': G_loss2.detach().item(),
                       'G2_state_dict': G2.state_dict(),
                       'generated_tensor2': generated_tensor2.detach(),
                       }

            manager(package)

            if opt.debug:
                break

        if epoch % opt.epoch_save == 0:
            torch.save(G1.state_dict(), os.path.join(test_opt.model_dir, "{:d}_G1.pt".format(epoch)))
            torch.save(G2.state_dict(), os.path.join(test_opt.model_dir, "{:d}_G2.pt".format(epoch)))
            image_dir = os.path.join(test_opt.image_dir, "{:d}".format(epoch))
            os.makedirs(image_dir, exist_ok=True)
            with torch.no_grad():
                for i, (input, target) in enumerate(test_data_loader):
                    input = input.to(DEVICE)
                    fake1 = G1(input)
                    fake2 = G2(input)
                    manager.save_image(fake1, os.path.join(image_dir, "{:d}_fake1.png".format(i)))
                    manager.save_image(fake2, os.path.join(image_dir, "{:d}_fake2.png".format(i)))
                    manager.save_image(target, os.path.join(image_dir, "{:d}_real.png".format(i)))

        if epoch > opt.epoch_decay:
            lr1 = update_lr(opt.lr, lr1, opt.n_epochs - opt.epoch_decay, G_optim1)
            lr2 = update_lr(opt.lr, lr2, opt.n_epochs - opt.epoch_decay, G_optim2)
    print("Total time taken: ", datetime.datetime.now() - start_time)
