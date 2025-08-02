import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureClassifier(nn.Module):
    def __init__(self, input_size=33*2, num_classes=3):
        super(GestureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Ejemplo de inicialización
if __name__ == "__main__":
    model = GestureClassifier()
    sample_input = torch.randn(1, 66)  # 33 keypoints x (x, y)
    output = model(sample_input)
    print("Output:", output)
