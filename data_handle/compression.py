import sys
import heapq

quant_bits = 4

def weights_analy(filename, recompression_func):
    data = read_file(filename)
    dic = num_cnt(data)
    code_dic = huffman(dic)
    tot_len = 0
    for i in range(2 ** quant_bits):
        tot_len += code_dic[i] * dic[i]
    tmpl = [dic[i] for i in range(2 ** quant_bits)]
    max_index = tmpl.index(max(tmpl))
    re_tot_len = recompression_func(data, max_index, code_dic, dic)
    print("hufman = {}, recompression = {}".format(tot_len, re_tot_len))
    return len(data) * quant_bits, tot_len, re_tot_len

def recompression1(data, num, code_dic, dic):
    data_len = len(data)
    max_bits  = 0
    interval = 0
    l = 0
    for i in range(1, data_len):
        if data[i] != num:
            interval = max(interval, i - l)
            l = i
    while (2 ** max_bits) < interval:
        max_bits += 1
    ans = sum([(code_dic[i] - 1) * dic[i] for i in range(2 ** quant_bits) if i != num])
    ans += sum([dic[i] * max_bits for i in range(2 ** quant_bits) if i != num])
    return ans + max_bits
    



def recompression(data, num, code_dic, dic):
    left_sum = sum([code_dic[i] * dic[i] for i in range(2 ** quant_bits) if i != num])
    data_len = len(data)
    max_bits = 0
    while (2 ** max_bits) < data_len:
        max_bits += 1
    l = 0
    i = 0
    while i < len(data):
        if data[i] != num:
            i += 1
            continue
        # print(i)
        l = i
        while i < len(data) and data[i] == num:
            i += 1
        if i - l + 1 > 2 * max_bits:
            left_sum += 2 * max_bits
        else:
            left_sum += i - l
    return left_sum + 2*max_bits + 8


def recompression2(data, num, code_dic, dic):
    left_sum = sum([code_dic[i] * dic[i] for i in range(2 ** quant_bits) if i != num])
    data_len = len(data)
    max_bits = 0
    l = 0
    i = 0
    interval = 0
    r = 0
    while i < len(data):
        if data[i] != num:
            i += 1
            continue
        r = l
        l = i
        while i < len(data) and data[i] == num:
            i += 1
        interval = max(interval, l - r)
        interval = max(interval, i - l)
        l = i
    while 2 ** max_bits < interval:
        max_bits += 1
    l = 0
    i = 0
    while i < len(data):
        if data[i] != num:
            i += 1
            continue
        # print(i)
        l = i
        while i < len(data) and data[i] == num:
            i += 1
        if i - l + 1 > 2 * max_bits:
            left_sum += 2 * max_bits
        else:
            left_sum += i - l
    return left_sum + 2*max_bits + 8




def num_cnt(data):
    ret_dic = {}
    for i in range(2 ** quant_bits):
        ret_dic[i] = 0
    for elem in data:
        ret_dic[elem] += 1
    return ret_dic

def huffman(dic):
    l = []
    ans = {}
    mark = {}
    for i in range(2 ** quant_bits):
        ans[i] = 0
        mark[i] = [i]
    for key in dic.keys():
        l.append((dic[key], int(key)))
    heapq.heapify(l)
    timestamp = 2 ** quant_bits
    while len(l) >= 2:
        u = heapq.heappop(l)
        v = heapq.heappop(l)
        for elem in mark[u[1]]:
            ans[elem] += 1
        for elem in mark[v[1]]:
            ans[elem] += 1
        heapq.heappush(l, (u[0] + u[1], timestamp))
        mark[timestamp] = mark[u[1]] + mark[v[1]]
        timestamp += 1
    return ans
    

def read_file(filename):
    with open(filename) as fd:
        data = fd.readlines()
        for i in range(len(data)):
            data[i] = int(float(data[i]) + 0.5)
        return data
def main():
    tot = 0
    tot_huffman = 0
    tot_recompression = 0
    func = [recompression, recompression1, recompression2]
    for i in range(2, len(sys.argv)):
        print(sys.argv[i])
        x, y, z= weights_analy(sys.argv[i], func[int(sys.argv[1])])
        tot += x
        tot_huffman += y
        tot_recompression += z
    print("origin = {}, hufman = {}, recompression = {}".format(tot, tot_huffman, tot_recompression))
    print("ratio1 = {}, ratio2 = {}".format(tot_huffman/ tot, tot_recompression / tot_huffman))
    print("final bits = {}, B = {}, KB = {}, MB = {}".format(tot_recompression, tot_recompression / 8, tot_recompression / (8 * 1024), tot_recompression / (8 * 1024 * 1024)))
    final_compression = tot_recompression / tot / (32 / quant_bits)
    print("final compression ratio = {}, 1 / ratio = {}".format(final_compression, 1 / final_compression))


main()
