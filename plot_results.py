import matplotlib.pyplot as plt

# These numbers come directly from your terminal output in the previous step
epochs = [1, 10, 20, 30, 40]
train_loss = [0.5330, 0.4393, 0.4105, 0.3929, 0.3760]
val_loss = [0.5128, 0.5196, 0.5154, 0.5081, 0.5141]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', color='red', marker='o')

plt.title('Training vs Validation Loss (MLP Churn Model)')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross Entropy Loss')
plt.legend()
plt.grid(True)

# Save the plot for your report
plt.savefig('loss_curve.png')
print("Graph saved as 'loss_curve.png'. You can now insert this into your report!")
plt.show()