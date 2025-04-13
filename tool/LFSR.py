class LFSR:
    def __init__(self, polynomial, initial_state):
        """
        初始化 LFSR
        :param polynomial: 本原多项式，表示为二进制列表（例如 [1, 0, 0, 1, 1] 表示 x^4 + x + 1）
        :param initial_state: 初始状态，表示为二进制列表（例如 [0, 0, 0, 1]）
        """
        self.polynomial = polynomial
        self.state = initial_state.copy()  # 初始化状态
        self.length = len(initial_state)  # LFSR 的级数

    def shift(self):
        """
        执行一次 LFSR 移位操作
        :return: 输出的比特
        """
        # 计算反馈值（异或所有参与反馈的位）
        feedback = 0
        for i in range(self.length):
            if self.polynomial[i]:
                feedback ^= self.state[i]

        # 输出最低位
        output = self.state[-1]

        # 右移状态
        self.state = [feedback] + self.state[:-1]

        return output

    def generate_sequence(self, num_bits):
        """
        生成伪随机序列
        :param num_bits: 需要生成的比特数
        :return: 生成的伪随机序列（列表）
        """
        sequence = []
        for _ in range(num_bits):
            sequence.append(self.shift())
        return sequence

    def contains_subarray(self, long_array, short_array):
        """
        检查长数组中是否包含连续的短数组
        :param long_array: 长度为 3000 的数组
        :param short_array: 长度为 250 的数组
        :return: 如果包含，返回 True；否则，返回 False
        """
        len_long = len(long_array)
        len_short = len(short_array)

        # 滑动窗口
        for i in range(len_long - len_short + 1):
            # 比较窗口内的元素
            if long_array[i:i + len_short] == short_array:
                print(i)
                return True
        return False


# 示例：使用本原多项式 x^4 + x + 1 构造 LFSR
if __name__ == "__main__":
    # # 本原多项式 x^4 + x + 1，表示为 [1, 0, 0, 1, 1]
    # L = 4
    # polynomial = [1, 0, 0, 1, 1]
    # initial_state = [1, 0, 0, 1]

    L = 13
    polynomial = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]  # 注意：多项式从高到低表示 L=13
    # 初始状态
    initial_state = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]

    # 创建 LFSR
    lfsr = LFSR(polynomial, initial_state)

    # 生成 15 位伪随机序列（最大周期为 2^4 - 1 = 15）
    sequence = lfsr.generate_sequence(2 ** L - 1)
    print("生成的伪随机序列长度:", len(sequence))
    print(sequence)
