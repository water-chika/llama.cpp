import argparse
import json


def parse_array(str):
    return json.loads(str)

def parse_params(params):
    params = params.split(',')
    results = {}
    prev_param = ''
    for param in params:
        param = param.split('=')
        if len(param) == 1:
            results[prev_param] += ',' + param[0]
        else:
            results[param[0]] = param[1]
            prev_param = param[0]
    for name in results:
        if results[name].startswith('['):
            results[name] = parse_array(results[name])
    return results

def type_size(str):
     type_size_dict = {
         'f16': 2,
         'f32': 4,
         'f64': 8,
         'bf16': 2,
         'mxfp4': 4.0/8.0,
     }
     if str in type_size_dict:
         return type_size_dict[str]
     elif str.startswith('q'):
        return float(str[1])/8.0
     elif str.startswith('iq'):
        return float(str[2])/8.0
     else:
         raise Exception('unknown type: ' + str)

def with_human_readable(f):
    if f > 1_000_000_000_000:
        return "{:.1f}T".format(f / 1_000_000_000_000)
    elif f > 1_000_000_000:
        return "{:.1f}G".format(f / 1_000_000_000)
    elif f > 1_000_000:
        return "{:.1f}M".format(f / 1_000_000)
    else:
        return f

def analyse_op_params(backend, results = {}):
    for op_perf in backend['op_perfs']:
        if op_perf['op'] not in results:
            results[op_perf['op']] = {}
        if op_perf['params'] not in results[op_perf['op']]:
            results[op_perf['op']][op_perf['params']] = {}
        value = 0
        if 'perf' in op_perf:
            value = op_perf['perf']
        if 'bandwidth' in op_perf:
            value = op_perf['bandwidth']
        results[op_perf['op']][op_perf['params']][backend['backend']] = value
    return results

def compare_op_params(data1, data2):
    results = analyse_op_params(data1['backends'][0])
    results = analyse_op_params(data2['backends'][0], results)

    theoretical_L3_read_bandwidth = 2_000_000_000_000
    theoretical_L3_write_bandwidth = 2_000_000_000_000
    L3_size = 32_000_000
    theoretical_read_bandwidth = 640_000_000_000
    theoretical_write_bandwidth = 640_000_000_000

    theoretical_f16_flops = 195_000_000_000_000
    theoretical_f32_flops = 48_000_000_000_000

    for op in results:
        for params in results[op]:
            if len(results[op][params]) == 2:
                results[op][params]['diff'] = "{:.1f}%".format((results[op][params]['ROCm0'] - results[op][params]['Vulkan0'])/results[op][params]['Vulkan0'] * 100)
            if op == 'MUL_MAT':
                try:
                    parsed_params = parse_params(params)
                    m = int(parsed_params['m'])
                    n = int(parsed_params['n'])
                    k = int(parsed_params['k'])
                    bs = parsed_params['bs']
                    nr = parsed_params['nr']
                    type_a = parsed_params['type_a']
                    type_b = parsed_params['type_b']
                    type_a_size = type_size(type_a)
                    type_b_size = type_size(type_b)
                    type_d_size = type_b_size

                    read_size = (m*k*type_a_size+k*n*type_b_size) * bs[0] * bs[1] * nr[0] * nr[1]
                    write_size = (m*n*type_d_size) * bs[0] * bs[1] * nr[0] * nr[1]
                    read_bandwidth = theoretical_read_bandwidth
                    write_bandwidth = theoretical_write_bandwidth
                    theoretical_bandwidth = min(read_bandwidth/(m*k*type_a_size+k*n*type_b_size), write_bandwidth/(m*n*type_d_size))
                    theoretical_flops = theoretical_f16_flops
                    if type_a == 'f32' and type_b == 'f32':
                        theoretical_flops = theoretical_f32_flops
                    results[op][params]['theoretical'] = min(float(2*m*k*n)*theoretical_bandwidth, theoretical_flops)
                    results[op][params]['diff_rocm_theoretical'] = "{:.1f}%".format((results[op][params]['ROCm0'] - results[op][params]['theoretical'])/results[op][params]['theoretical'] * 100)
                    results[op][params]['diff_vulkan_theoretical'] = "{:.1f}%".format((results[op][params]['Vulkan0'] - results[op][params]['theoretical'])/results[op][params]['theoretical'] * 100)
                except Exception as e:
                    print(e)
            elif op == 'CPY':
                try:
                    parsed_params = parse_params(params)
                    type_src = parsed_params['type_src']
                    type_dst = parsed_params['type_dst']
                    type_src_size = type_size(type_src)
                    type_dst_size = type_size(type_dst)
                    ne = parsed_params['ne']

                    theoretical_bandwidth_bound_op = min(theoretical_read_bandwidth / type_src_size, theoretical_write_bandwidth / type_dst_size)
                    results[op][params]['theoretical'] = theoretical_bandwidth_bound_op * (type_src_size + type_dst_size) / 1_000_000_000
                    results[op][params]['diff_rocm_theoretical'] = "{:.1f}%".format((results[op][params]['ROCm0'] - results[op][params]['theoretical'])/results[op][params]['theoretical'] * 100)
                    results[op][params]['diff_vulkan_theoretical'] = "{:.1f}%".format((results[op][params]['Vulkan0'] - results[op][params]['theoretical'])/results[op][params]['theoretical'] * 100)
                except Exception as e:
                    print(op, 'exception', e)
            elif op == 'IM2COL':
                try:
                    parsed_params = parse_params(params)
                    type_input = parsed_params['type_input']
                    type_dst = parsed_params['dst_type']
                    type_input_size = type_size(type_input)
                    type_dst_size = type_size(type_dst)
                    ne_input = parsed_params['ne_input']
                    ne_kernel = parsed_params['ne_kernel']

                    theoretical_bandwidth_bound_op = min(theoretical_read_bandwidth / type_input_size, theoretical_write_bandwidth / (type_dst_size * ne_kernel[0] * ne_kernel[1]))
                    results[op][params]['theoretical'] = theoretical_bandwidth_bound_op * (type_input_size + type_dst_size * ne_kernel[0] * ne_kernel[1]) / 1_000_000_000
                    results[op][params]['diff_rocm_theoretical'] = "{:.1f}%".format((results[op][params]['ROCm0'] - results[op][params]['theoretical'])/results[op][params]['theoretical'] * 100)
                    results[op][params]['diff_vulkan_theoretical'] = "{:.1f}%".format((results[op][params]['Vulkan0'] - results[op][params]['theoretical'])/results[op][params]['theoretical'] * 100)
                except Exception as e:
                    print(op, 'exception', e)
    
    for op in results:
        for params in results[op]:
            if 'Vulkan0' in results[op][params]:
                results[op][params]['Vulkan0'] = with_human_readable(results[op][params]['Vulkan0'])
            if 'ROCm0' in results[op][params]:
                results[op][params]['ROCm0'] = with_human_readable(results[op][params]['ROCm0'])
            if 'theoretical' in results[op][params]:
                results[op][params]['theoretical'] = with_human_readable(results[op][params]['theoretical'])
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file1')
    parser.add_argument('json_file2')
    parser.add_argument('output_file')

    args = parser.parse_args()

    with open(args.json_file1, 'r') as json_file1:
        backend1 = json.load(json_file1)
        with open(args.json_file2, 'r') as json_file2:
            backend2 = json.load(json_file2)

            results = compare_op_params(backend1, backend2)
            with open(args.output_file, 'w') as output:
                json.dump(results, output, indent=4)