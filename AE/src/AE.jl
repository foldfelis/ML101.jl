module AE

include("MNIST/MNIST.jl")

function get_data()
    # load full training set
    train_x, train_y = MNIST.traindata()

    # load full test set
    test_x,  test_y  = MNIST.testdata()

    return train_x, train_y, test_x, test_y
end

end
