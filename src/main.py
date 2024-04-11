import torch

if __name__ == '__main__':

  if torch.cuda.is_available():
    print(f'Running on GPU: {torch.cuda.get_device_name()}.')
  else:
    print('Running on CPU.')
