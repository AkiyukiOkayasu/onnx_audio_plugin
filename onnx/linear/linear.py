import torch

class Net(torch.nn.Module):
  def __init__(self, n_input, n_output):
    super().__init__()

    self.l1 = torch.nn.Linear(n_input, n_output)
    torch.nn.init.constant_(self.l1.weight, -1.0)
    torch.nn.init.constant_(self.l1.bias, 0.0)

  def forward(self, x):
    x1 = self.l1(x)
    return x1

def main():
    torch.manual_seed(123)
    # モデルのインスタンスを作成
    n_input = 1
    n_output = 1
    model = Net(n_input, n_output)
    # ダミーデータを作成
    input_tensor = torch.randn(n_input)
    # モデルの推論を実行
    output_tensor = model(input_tensor)
    print("Input Tensor:", input_tensor)
    print("Output Tensor:", output_tensor)
    
    print(model.l1)
    model.eval()
    dummy_input = torch.randn(1,)
    torch.onnx.export(model, dummy_input, "linear.onnx", input_names=["input"], dynamo=False)

if __name__ == "__main__":
    main()
