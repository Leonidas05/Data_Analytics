import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Datos de ejemplo (normalizados)
x_train = torch.tensor([5, 15, 25, 35, 45, 55], dtype=torch.float32).view(-1, 1) / 55
y_train = torch.tensor([5, 20, 14, 32, 22, 38], dtype=torch.float32).view(-1, 1) / 38

# Inicializar pesos y bias (pendiente e intersección)
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Configurar el optimizador
learning_rate = 0.01
optimizer = optim.SGD([w, b], lr=learning_rate)

# Configurar la gráfica
fig, ax = plt.subplots()
plt.scatter(x_train.numpy(), y_train.numpy(), label='Original data', color='blue')       
line, = ax.plot(x_train.numpy(), x_train.numpy() * w.item() + b.item(), 'r-', label='Fitted line')
                
# Parámetros para el criterio de parada
tolerancia = 1e-6
loss_previo = float('inf')

# Función de actualización para la animación
def update(i):
    global loss_previo
    optimizer.zero_grad()
    y_pred = x_train * w + b
    loss = ((y_pred - y_train) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Actualizar la línea de la gráfica
    line.set_ydata(x_train.numpy() * w.item() + b.item())

    # Imprimir los valores actuales de w, b y pérdida
    print(f"Epoch {i}: w = {w.item():.3f}, b = {b.item():.3f}, Loss = {loss.item():.4f}")

    # Chequear la convergencia
    if abs(loss_previo - loss.item()) < tolerancia:
        print(f"Convergencia alcanzada en la época {i}.")
        ani.event_source.stop() # Intentar detener la animación
    loss_previo = loss.item()
    # Actualizar el título de la gráfica con el último valor de pérdida

    ax.set_title(f'Epoch {i}:w Loss: {loss.item():.4f}')
    return line,

# Crear animación
ani = FuncAnimation(fig, update, frames=range(1000), blit=False, interval=10, repeat=False)

plt.legend()
plt.show()