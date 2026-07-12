import numpy as np

from mlab.neural_networks._cnn import (
    CNNClassifier,
    ConvLayer,
    PoolingLayer,
    ReLULayer,
    SoftmaxLayer,
    ModularLinearLayer,
)


def test_layers():
    np.random.seed(42)

    X = np.random.randn(2, 8, 8, 1)

    conv = ConvLayer(
        in_channels=1,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1
    )

    conv_output = conv(X)
    assert conv_output.shape == (2, 8, 8, 4)

    conv_gradient = np.ones_like(conv_output)
    input_gradient = conv.backward(conv_gradient)

    assert input_gradient.shape == X.shape
    assert conv.grad_weight.shape == (4, 1, 3, 3)
    assert conv.grad_bias.shape == (4,)

    pool = PoolingLayer(pool_size=2, stride=2)
    pool_output = pool(conv_output)

    assert pool_output.shape == (2, 4, 4, 4)

    pool_gradient = np.ones_like(pool_output)
    pooled_input_gradient = pool.backward(pool_gradient)

    assert pooled_input_gradient.shape == conv_output.shape

    relu = ReLULayer()
    relu_output = relu(np.array([[-2.0, 0.0, 3.0]]))
    assert np.array_equal(relu_output, np.array([[0.0, 0.0, 3.0]]))

    softmax = SoftmaxLayer()
    probabilities = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert probabilities.shape == (1, 3)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)

    linear = ModularLinearLayer(input_size=5, output_size=3)
    linear_output = linear(np.ones((2, 5)))
    assert linear_output.shape == (2, 3)

    print("Layer tests passed.")


def test_cnn_classifier():
    np.random.seed(42)

    n_samples = 60
    X = np.random.rand(n_samples, 8, 8, 1)

    y = np.zeros(n_samples, dtype=int)

    # Class 1: bright upper-left square
    X[20:40, 0:4, 0:4, 0] += 2.0
    y[20:40] = 1

    # Class 2: bright lower-right square
    X[40:60, 4:8, 4:8, 0] += 2.0
    y[40:60] = 2

    model = CNNClassifier(
        input_shape=(8, 8, 1),
        num_classes=3,
        lr=0.05,
        epochs=30,
        batch_size=10,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (n_samples,)
    assert np.all(predictions >= 0)
    assert np.all(predictions < 3)
    assert len(model.layers_) == 5
    assert len(model.loss_curve_) > 0

    accuracy = np.mean(predictions == y)

    print("CNN training test passed.")
    print(f"Training accuracy: {accuracy:.2%}")
    print(f"First loss: {model.loss_curve_[0]:.6f}")
    print(f"Last loss: {model.loss_curve_[-1]:.6f}")


if __name__ == "__main__":
    test_layers()
    test_cnn_classifier()
    print("\nAll CNN tests passed.")