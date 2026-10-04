import torch
import torch.nn as nn
import numpy as np
import os
FWD_DIR = 'forward_7p_40_sig_weights'
INV_FILE = 'PINN_40sig_7p.pt'
# --- 1. АРХИТЕКТУРЫ СЕТЕЙ ---
class ForwardSurrogate(nn.Module):
    def __init__(self):
        super(ForwardSurrogate, self).__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 40)
        self.gelu = nn.GELU()
        
    def load_txt_weights(self):
        def assign_layer(layer, w_file, b_file):
            if os.path.exists(w_file) and os.path.exists(b_file):
                W = np.loadtxt(w_file, ndmin=2)
                b = np.loadtxt(b_file)
                layer.weight.data = torch.FloatTensor(W.T)
                layer.bias.data = torch.FloatTensor(b)
        assign_layer(self.fc1, os.path.join(FWD_DIR, 'W1.txt'), os.path.join(FWD_DIR, 'b1.txt'))
        assign_layer(self.fc2, os.path.join(FWD_DIR, 'W2.txt'), os.path.join(FWD_DIR, 'b2.txt'))
        assign_layer(self.fc3, os.path.join(FWD_DIR, 'W3.txt'), os.path.join(FWD_DIR, 'b3.txt'))
        assign_layer(self.fc4, os.path.join(FWD_DIR, 'W4.txt'), os.path.join(FWD_DIR, 'b4.txt'))
        assign_layer(self.fc5, os.path.join(FWD_DIR, 'W5.txt'), os.path.join(FWD_DIR, 'b5.txt'))
class InverseNet40CH(nn.Module):
    def __init__(self):
        super(InverseNet40CH, self).__init__()
        self.fc1 = nn.Linear(40, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 7)
        self.gelu = nn.GELU()
# --- 2. ФУНКЦИЯ ЭКСПОРТА В СИ ---
def export_to_c_header(model, filename, prefix, requires_backprop=False):
    print(f'Экспорт {prefix} в {filename}...')
    with open(filename, 'w') as f:
        f.write(f"#ifndef {prefix}_WEIGHTS_H\n#define {prefix}_WEIGHTS_H\n\n")
        for name, param in model.named_parameters():
            name_clean = name.replace('.', '_')
            flat_data = param.detach().cpu().numpy().flatten()
            array_name = f"{prefix}_{name_clean}"
            f.write(f"// Type: {name}, Shape: {list(param.shape)}\n")
            f.write(f"const float {array_name}[{len(flat_data)}] = {{\n    ")
            lines = []
            for i in range(0, len(flat_data), 8):
                chunk = flat_data[i:i+8]
                lines.append(", ".join([f"{val:.6e}f" for val in chunk]))
            f.write(",\n    ".join(lines))
            f.write("\n};\n\n")
            
            if requires_backprop and 'weight' in name:
                transposed = param.detach().cpu().numpy().T
                flat_t = transposed.flatten()
                array_name_t = f"{prefix}_{name_clean}_T"
                f.write(f"// TRANSPOSED Type: {name}, Shape: {list(transposed.shape)}\n")
                f.write(f"const float {array_name_t}[{len(flat_t)}] = {{\n    ")
                lines_t = []
                for i in range(0, len(flat_t), 8):
                    chunk = flat_t[i:i+8]
                    lines_t.append(", ".join([f"{val:.6e}f" for val in chunk]))
                f.write(",\n    ".join(lines_t))
                f.write("\n};\n\n")
        f.write(f"#endif // {prefix}_WEIGHTS_H\n")
if __name__ == '__main__':
    print("Инициализация сетей...")
    device = torch.device('cpu') # Нам нужен только экспорт, CPU достаточно
    
    fwd = ForwardSurrogate()
    fwd.load_txt_weights()
    
    inv = InverseNet40CH()
    inv.load_state_dict(torch.load(INV_FILE, map_location=device, weights_only=True))
    
    export_to_c_header(fwd, 'forward_weights.h', 'FWD', requires_backprop=True)
    export_to_c_header(inv, 'inverse_weights.h', 'INV', requires_backprop=False)
    
    print("\nГотово! Файлы .h успешно созданы в текущей папке.")