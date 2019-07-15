set -ex
python test.py --dataroot ./datasets/LFW/testB --name sketch_pic --model pix2pix --which_model_netG unet_256  --dataset_mode aligned --norm batch
