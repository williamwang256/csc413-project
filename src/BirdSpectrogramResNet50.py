import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.models as models

# ResNet50
class BirdSpectrogramResNet50(nn.Module):
  def __init__(self, num_species):
    super().__init__()

    self.network = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # modify last layer
    num_ftrs = self.network.fc.in_features # fc is the fully connected last layer
    self.network.fc = nn.Linear(num_ftrs, num_species) # MAKE SURE TO UPDATE THIS TO THE CORRECT NUMBER OF BIRDS
    nn.init.xavier_uniform_(self.network.fc.weight) # initialize weights

  def forward(self, xb):
    return self.network(xb)

def accuracy(model, dl, device):
  model.to(device)
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for X, t in dl:
      X = X.to(device)
      t = t.to(device)

      z = model(X)
      pred = z.max(1, keepdim=True)[1] # index of max log-probability
      correct += pred.eq(t.view_as(pred)).sum().item()
      total += X.shape[0]

  return correct / total

def train_model(model,
                device,
                train_ds,
                val_ds,
                learning_rate=0.001,
                batch_size=64,
                epochs=10,
                plot_every=10,
                plot=True):

  train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
  val_dl   = torch.utils.data.DataLoader(val_ds, 256)

  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate)

  # for plotting
  iters, train_loss, train_acc, val_acc = [], [], [], []
  iter_count = 0 # count the number of iterations that has passed

  for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch+1, epochs))

    # put model in train mode

    for X, t in train_dl:
      X = X.to(device)
      t = t.to(device)

      model.train()
      z = model(X)
      loss = criterion(z, t)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      iter_count += 1
      if iter_count % plot_every == 0:
        loss = float(loss)
        t_acc = accuracy(model, train_dl, device)
        v_acc = accuracy(model, val_dl, device)

        print("Iter {}; Loss {}; Train Acc {}; Val Acc {}".format(iter_count, loss, t_acc, v_acc))

        iters.append(iter_count)
        train_loss.append(loss)
        train_acc.append(t_acc)
        val_acc.append(v_acc)

    print()

  # plot result
  if plot:
    plt.figure()
    plt.plot(iters[:len(train_loss)], train_loss)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(iters[:len(train_acc)], train_acc)
    plt.plot(iters[:len(val_acc)], val_acc)
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig("acc.png")