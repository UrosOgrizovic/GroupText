import matplotlib.pyplot as plt


def plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss, num_rows):
    """
    plots training and validation curves
    :param train_acc:
    :param val_acc:
    :param train_loss:
    :param val_loss:
    :param num_rows:
    :return:
    """
    epochs = range(1, len(train_acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, train_acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.xlabel('Epochs')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.savefig('vanilla_cnn_'+num_rows+'_train_val_acc', bbox_inches='tight')

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('vanilla_cnn_'+num_rows+'_train_val_loss', bbox_inches='tight')

    plt.show()