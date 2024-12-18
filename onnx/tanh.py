import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from scipy.fft import fft

# プロジェクトの概要:
# このプロジェクトは、ReLUを使用してONNXモデルを生成し、tanh関数を模倣するニューラルネットワークを訓練する例です。
# モデルは、入力範囲を-5から5にマッピングし、tanh関数を学習します。

# 再現性のためのランダムシードの設定
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_hidden)
        self.l4 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.l4(x)
        return x

def main():
    # ハイパーパラメータ
    n_input = 1
    n_hidden = 10
    n_output = 1
    num_epochs = 15000
    learning_rate = 0.002
    seed = 12345  # 任意のシード値

    # ランダムシードの設定
    set_seed(seed)

    # モデルのインスタンスを作成
    model = Net(n_input, n_hidden, n_output)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # トレーニングデータの作成
    x_train = np.linspace(-5, 5, 10000).reshape(-1, 1).astype(np.float32)
    y_train = np.tanh(x_train)

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)

    # モデルの訓練
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}')

    # モデルの推論を実行
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[0.0]])
        test_output = model(test_input)
        print("Test Input:", test_input.item())
        print("Test Output:", test_output.item())

    # テストデータでの評価
    x_test = np.linspace(-5, 5, 100).reshape(-1, 1).astype(np.float32)
    y_test = np.tanh(x_test)
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(y_test)

    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.8f}')

    # 周波数領域での誤差評価
    y_test_fft = fft(y_test_tensor.numpy().flatten())
    test_outputs_fft = fft(test_outputs.numpy().flatten())
    freq_error = np.mean(np.abs(y_test_fft - test_outputs_fft))
    print(f'Frequency Domain Error: {freq_error:.8f}')

    # モデルをONNX形式でエクスポート
    dummy_input = torch.randn(1, 1)
    torch.onnx.export(model, dummy_input, "tanh.onnx", input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

if __name__ == "__main__":
    main()
