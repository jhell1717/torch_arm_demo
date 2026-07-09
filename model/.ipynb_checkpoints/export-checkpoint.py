import torch
from train import DragMLP

def main():
    model = DragMLP()
    model.load_state_dict(torch.load("/Users/joshuahellewell/arm-project/model/drag_mlp.pt"))
    model.eval()

    dummy_input = torch.randn(1, 4)  # batch of 1, 4 geometry params

    torch.onnx.export(
        model,
        dummy_input,
        "/Users/joshuahellewell/arm-project/model/drag_mlp.onnx",
        input_names=["geometry_params"],
        output_names=["drag_coefficient"],
        dynamic_axes={"geometry_params": {0: "batch"}, "drag_coefficient": {0: "batch"}},
        opset_version=17,
    )
    print("Exported model/drag_mlp.onnx")

if __name__ == "__main__":
    main()