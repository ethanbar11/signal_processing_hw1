import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BOTTOM_LIMIT = 0
TOP_LIMIT = 255
EPSILON = 10


class Quantizer:
    def quantize_val(self, param, min_val, max_val):
        # Should be implemented in son.
        pass



    def quantize(self, img):
        min_val = np.min(img)
        max_val = np.max(img)
        new_img = []
        for i in range(len(img)):
            new_img.append(self.quantize_val(img[i], min_val, max_val))
        return new_img


class Uniform_Quantizer(Quantizer):
    def __init__(self, bits_amount):
        self.bits_amount = bits_amount

    def get_decision_levels(self, min_val, max_val):
        delta = (max_val - min_val) * 1.0 / np.power(2, self.bits_amount)
        decision_levels = []
        for i in range(2 ** self.bits_amount):
            d_i = min_val + i * delta
            decision_levels.append(d_i)
        return decision_levels

    def get_representation_levels(self, min_val, max_val):
        delta = (max_val - min_val) * 1.0 / np.power(2, self.bits_amount)
        representation_levels = []
        for i in range(2 ** self.bits_amount):
            d_i = min_val + (i - 1 / 2) * delta
            representation_levels.append(d_i)
        return representation_levels

    def quantize_val(self, x, min_val, max_val):
        delta = (max_val - min_val) * 1.0 / np.power(2, self.bits_amount)
        d_i = delta * np.floor((x - min_val) / delta)
        r_i = min_val + d_i + delta / 2
        return r_i


class MaxFloydQuantizer:
    def __init__(self, bits_amount):
        self.bits_amount = bits_amount
        self.uniform_quantizer = Uniform_Quantizer(bits_amount)
        self.initial_dec = self.uniform_quantizer.get_decision_levels(BOTTOM_LIMIT, TOP_LIMIT)
        self.epsilon = EPSILON
        self.decision_levels = None
        self.representation_levels = None

    def init(self, pdf):
        improvement = np.inf
        last_err = np.inf
        dec = self.initial_dec
        rep_length = len(dec) + 1
        rep = np.array([0 for i in range(rep_length)])
        while improvement > epsilon:
            # Update representation levels.
            self.max_floyd_update_representation_levels(dec, pdf, rep)
            self.max_floyd_update_decision_levels(dec, rep)
            err, improvement = self.calculate_error_and_improvement(dec, last_err, pdf, rep)
            last_err = err
        self.decision_levels = dec
        self.representation_levels = rep

    def calculate_error_and_improvement(self, dec, last_err, pdf, rep):
        err = 0
        for i in range(len(dec)):
            bottom = int(dec[i - 1] if i > 0 else BOTTOM_LIMIT)
            top = int(dec[i]) if i < len(dec) else TOP_LIMIT
            for j in range(bottom, top + 1):
                err += (j - rep[i]) ** 2 * pdf[j]
        if last_err != np.inf:
            improvement = last_err - err
        else:
            improvement = epsilon * 2
        return err, improvement

    def max_floyd_update_decision_levels(self, dec, rep):
        # Update Decision levels.
        for i in range(len(dec)):
            dec[i] = round((rep[i] + rep[i + 1]) / 2.0) if i < (len(rep) - 1) else TOP_LIMIT

    def max_floyd_update_representation_levels(self, dec, pdf, rep):
        for i in range(len(rep)):
            bottom = int(dec[i - 1] if i > 0 else BOTTOM_LIMIT)
            top = int(dec[i]) if i < len(dec) else TOP_LIMIT
            dividend = 0
            divisor = 0
            for j in range(bottom, top + 1):
                dividend += j * pdf[j]
                divisor += pdf[j]
            rep[i] = dividend / divisor


def mean_squared_error(img, q_img):
    error = np.sum(np.power((img - q_img), 2))
    error /= len(img)
    return error


def show_histogram(img):
    hist = np.bincount(img, minlength=TOP_LIMIT + 1)
    # plt.plot(hist)
    # plt.show()
    return hist


if __name__ == '__main__':
    # Exe 1

    # We will use flattened image in this part.
    img = np.array(Image.open('lena.jpg')).astype(np.int64).flatten()
    hist = show_histogram(img)

    # Exe 2.a
    # errors = []
    # for bits_amount in range(1, 9):
    #     quantizer = Uniform_Quantizer(bits_amount)
    #     new_img = quantizer.quantize(img)
    #     mse = mean_squared_error(img, new_img)
    #     errors.append(mse)
    #     print('Bits amount {}, error : {}'.format(bits_amount, mse))
    # plt.xlabel('Bits amount')
    # plt.ylabel('Error')
    # plt.xticks(np.arange(8), np.arange(1, 9))
    # plt.plot(errors)
    # plt.show()

    # Exe 2.b
    # for i in range(1, 4):
    #     bits_amount = 2 ** i
    #     quantizer = Uniform_Quantizer(bits_amount)
    #     decisions = quantizer.get_decision_levels(np.min(img), np.max(img))
    #     represents = quantizer.get_representation_levels(np.min(img), np.max(img))
    # plt.plot(decisions, 'bo')
    # plt.xticks(np.arange(2 ** bits_amount), np.arange(1, 2 ** (bits_amount + 1)))
    # plt.xlabel('Bit Amount')
    # plt.ylabel('Decision Levels')
    # plt.show()

    # plt.cla()
    # plt.xticks(np.arange(2 ** bits_amount), np.arange(1, 2 ** (bits_amount + 1)))
    # plt.xlabel('Bit Amount')
    # plt.ylabel('Representation Levels')
    # plt.plot(represents, 'bo')
    # plt.show()
    # plt.cla()

    # Exe 3
    epsilon = 10
    for bits_amount in range(4, 9):
        quantizer = MaxFloydQuantizer(bits_amount)
        dec_lvls, rep_lvls = max_floyd(hist, initial_decisions, epsilon)

    # plt.plot(decisions, 'bo')
    # plt.xticks(np.arange(2 ** bits_amount), np.arange(1, 2 ** (bits_amount + 1)))
    # plt.xlabel('Bit Amount')
    # plt.ylabel('Decision Levels')
    # plt.show()
