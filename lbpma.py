import pandas as pd
import pm4py
import numpy as np
import requests
import json
import re
from pm4py import PetriNet
import os

# 读取日志文件
log = pd.read_csv("../event log/2.csv", encoding='ANSI')
df = pm4py.format_dataframe(log, activity_key='Activity', case_id='CaseID', timestamp_key='Starttimes')

# 发现Petri网
net, im, fm = pm4py.discover_petri_net_inductive(df, activity_key='Activity', case_id_key='CaseID', timestamp_key='Starttimes')

# 获取变迁标签
transition_labels = [t.label for t in net.transitions if t.label is not None]

num_otransitions = len(net.transitions)
print("Number of transitions:", num_otransitions)
num_oplaces = len(net.places)
print("Number of places:", num_oplaces)
num_oarcs = len(net.arcs)
print("Number of arcs:", num_oarcs)

# 语义相关性缓存
semantic_cache = {}

def get_semantic_relevance(label1, label2):
    # 先检查缓存
    if (label1, label2) in semantic_cache:
        return semantic_cache[(label1, label2)]
    if (label2, label1) in semantic_cache:
        return semantic_cache[(label2, label1)]

    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": """You are a helpful coding assistant. Respond with only a single number between 0 and 1 indicating the semantic relevance between the two given labels, without any additional text."
                                          Here are some examples to guide your response:
                                          1. The semantic relevance between 'CSDN上学习Java对SQL增删改查' and '学习多态知识' is 0.8.
                                          2. The semantic relevance between '编写Testusers类' and '编写SetPow函数' is 0.8.
                                          3. The semantic relevance between '组建团队分工' and '建立Teacher类' is 0.2.
                                          4. The semantic relevance between '需求分析与测试' and '开发与测试' is 0.4.
                                          5. The semantic relevance between '学习与技术调研' and '开发与测试' is 0.4.
                                          6. The semantic relevance between '编写教师一部分代码' and '团队交流' is 0.4.
                                          2. The semantic relevance between '编写函数代码' and '上网查资料学习' is 0.3.
                                          Based on these examples, calculate the semantic relevance between the two new labels and return only a single number.
                                          """},
            {"role": "user", "content": f"In the context of a software development project, calculate the semantic relevance between '{label1}' and '{label2}' and only return a specific number without any additional text."}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    data = response.json()
    content = data['choices'][0]['message']['content'].strip()

    # 使用正则表达式提取数值部分
    match = re.search(r"[-+]?\d*\.?\d+", content)
    if match:
        relevance = float(match.group())
    else:
        raise ValueError(f"Could not extract a valid number from the response: {content}")

    # 缓存结果
    semantic_cache[(label1, label2)] = relevance
    semantic_cache[(label2, label1)] = relevance

    return relevance

def calculate_semantic_relevance_batch(labels):
    n = len(labels)
    relevance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                relevance_matrix[i][j] = 1.0
            else:
                relevance = get_semantic_relevance(labels[i], labels[j])
                relevance_matrix[i][j] = relevance
                relevance_matrix[j][i] = relevance
    return relevance_matrix

# 新增函数：尝试从文件加载或计算语义相关性矩阵
def load_or_calculate_relevance_matrix(transition_labels):
    filename = "semantic_relevance_matrix.npy"

    if os.path.exists(filename):
        try:
            relevance_matrix = np.load(filename)
            print(f"Relevance matrix loaded from '{filename}'.")
            return relevance_matrix
        except Exception as e:
            print(f"Error loading relevance matrix: {e}")

    print("Calculating relevance matrix...")
    relevance_matrix = calculate_semantic_relevance_batch(transition_labels)
    np.save(filename, relevance_matrix)
    print(f"Relevance matrix calculated and saved to '{filename}'.")

    return relevance_matrix

# 计算或加载初始Petri网的语义相关性矩阵
relevance_matrix = load_or_calculate_relevance_matrix(transition_labels)

def get_predecessors(net, t):
    return [arc.source for arc in net.arcs if arc.target == t]


def get_successors(net, t):
    return [arc.target for arc in net.arcs if arc.source == t]


def check_sequence_structure(net, t1, t2):
    t1_successors = get_successors(net, t1)
    t2_predecessors = get_predecessors(net, t2)
    return any(p in t1_successors for p in t2_predecessors)


def is_enabled(transition, marking):
    """检查变迁是否在给定标识下启用"""
    input_places = [arc.source for arc in net.arcs if arc.target == transition]
    return all(marking.get(place, 0) > 0 for place in input_places)


def apply_transition(net, marking, transition):
    """应用变迁并返回新的标识"""
    new_marking = marking.copy()
    for arc in net.arcs:
        if arc.source == transition:
            if arc.target in new_marking:
                new_marking[arc.target] += 1
            else:
                new_marking[arc.target] = 1

            if arc.source in new_marking:
                new_marking[arc.source] -= 1
            else:
                new_marking[arc.source] = -1

    return new_marking



def check_concurrent_structure(net, t1, t2, marking):
    """根据定义检查变迁 t1 和 t2 是否在标识 marking 下并发"""
    if not (is_enabled(t1, marking) and is_enabled(t2, marking)):
        return False

    # 应用 t1 和 t2
    marking_after_t1 = apply_transition(net, marking, t1)
    marking_after_t2 = apply_transition(net, marking, t2)

    # 检查 t2 是否可以在 marking_after_t1 之后触发
    if is_enabled(t2, marking_after_t1):
        # 应用 t2 并检查 t1 是否可以在 marking_after_t2 之后触发
        marking_after_t2_after_t1 = apply_transition(net, marking_after_t2, t2)
        if is_enabled(t1, marking_after_t2_after_t1):
            return True

    return False


initial_marking = {place: 1 for place in net.places}  # 假设所有初始标记都为 1


def check_loop_structure(net, t1, t2):
    t1_successors = get_successors(net, t1)
    t2_successors = get_successors(net, t2)
    return any(get_successors(net, s) == t2 for s in t1_successors) and \
        any(get_successors(net, s) == t1 for s in t2_successors)


def can_trigger(t1, t2, marking, net):
    """检查在给定标识下，t1 和 t2 是否可以同时触发"""
    return is_enabled(t1, marking) and is_enabled(t2, marking)


def check_branch_structure(net, t1, t2, marking):
    t1_predecessors = get_predecessors(net, t1)
    t2_predecessors = get_predecessors(net, t2)
    return bool(set(t1_predecessors) & set(t2_predecessors)) and can_trigger(t1, t2, marking, net)


def can_aggregate(net, t1, t2):
    if check_sequence_structure(net, t1, t2):
        return True
    if check_concurrent_structure(net, t1, t2, initial_marking):
        return True
    if check_loop_structure(net, t1, t2):
        return True
    return False


def generate_structure_matrix(net, transitions):
    n = len(transitions)
    structure_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                if can_aggregate(net, transitions[i], transitions[j]):
                    structure_matrix[i][j] = 1
                elif check_branch_structure(net, transitions[i], transitions[j]):
                    structure_matrix[i][j] = 0
    return structure_matrix



# 生成结构性矩阵
print("Generating structure matrix...")
transitions_list = [t for t in net.transitions if t.label is not None]
structure_matrix = generate_structure_matrix(net, transitions_list)
print("Structure matrix generated.")


def generate_aggregated_label(label):
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system",
             "content": "You are a helpful assistant who creates concise and meaningful names for merged transitions in a process model."},
            {"role": "user",
             "content": f"Given that '{label}' is an activities from a software development project, map the aggregated activity to the most appropriate category from the following list:'组建团队分工', '需求分析与设计', '学习与技术调研', '开发与测试', '团队交流', '做课件'. Provide only the category name."}
        ],
        "temperature": 0.7,
        "max_tokens": 10,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    data = response.json()
    content = data['choices'][0]['message']['content'].strip()

    # 提取生成的名称并返回
    return content


def is_place_critical(net, place):
    """
    检查库所是否在保持Petri网连通性方面起到关键作用。
    """
    incoming_arcs = [arc for arc in net.arcs if arc.target == place]
    outgoing_arcs = [arc for arc in net.arcs if arc.source == place]

    if not incoming_arcs or not outgoing_arcs:
        return True  # 如果库所没有进入或离开边，它可能是关键的

    connected_transitions = set(arc.source for arc in outgoing_arcs) | set(arc.target for arc in incoming_arcs)
    if len(connected_transitions) > 1:
        return True  # 如果库所连接多个变迁，它可能是关键的

    return False

def remove_unused_places_and_arcs(net, merged_transitions):
    places_to_remove = set()
    arcs_to_remove = set()

    for place in net.places:
        incoming_transitions = [arc.source for arc in net.arcs if arc.target == place]
        outgoing_transitions = [arc.target for arc in net.arcs if arc.source == place]

        if len(incoming_transitions) == 1 and len(outgoing_transitions) == 1:
            t1 = incoming_transitions[0]
            t2 = outgoing_transitions[0]

            if t1.label == t2.label or (
                    t1 in merged_transitions and t2 in merged_transitions and merged_transitions[t1] ==
                    merged_transitions[t2]):
                if not is_place_critical(net, place):
                    places_to_remove.add(place)
                    arcs_to_remove.update([arc for arc in net.arcs if arc.source == place or arc.target == place])

    for place in places_to_remove:
        net.places.remove(place)
    for arc in arcs_to_remove:
        net.arcs.remove(arc)

    return net
def remove_duplicate_arcs(net):
    """
    删除Petri网中库所和变迁之间的重复弧。
    """
    seen_arcs = set()
    arcs_to_remove = set()

    for arc in net.arcs:
        if (arc.source, arc.target) in seen_arcs:
            arcs_to_remove.add(arc)
        else:
            seen_arcs.add((arc.source, arc.target))

    for arc in arcs_to_remove:
        net.arcs.remove(arc)

    return net
#
#
def remove_redundant_end_places(net):
    for transition in net.transitions:
        # 获取没有出边的库所
        end_places = [place for place in get_successors(net, transition) if len(get_successors(net, place)) == 0]

        if len(end_places) > 1:

            # 保留一个库所，删除多余的库所
            for place in end_places[1:]:
                net.places.remove(place)
                for arc in list(net.arcs):
                    if arc.source == transition and arc.target == place:
                        net.arcs.remove(arc)

    return net


def aggregate_transitions(net, similarity_matrix, structure_matrix, threshold, final_marking):
    transitions = [t for t in net.transitions if t.label is not None]
    merged_transitions = {}  # 用于存储聚合信息
    iteration = 0

    while True:
        candidate_pairs = []

        current_transitions = [t for t in transitions]

        for i in range(len(current_transitions)):
            for j in range(i + 1, len(current_transitions)):
                if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1]:
                    if similarity_matrix[i][j] >= threshold and structure_matrix[i][j] == 1:
                        candidate_pairs.append((i, j))

        if not candidate_pairs:
            print("No more transitions can be aggregated.")
            break

        to_remove = set()

        for i, j in candidate_pairs:
            if i in to_remove or j in to_remove:
                continue

            t1 = current_transitions[i]
            t2 = current_transitions[j]
            # 将两个标签合并为一个字符串
            combined_label = f"{t1.label} + {t2.label}"
            new_label = generate_aggregated_label(combined_label)

            new_transition = PetriNet.Transition(new_label, new_label)
            net.transitions.add(new_transition)

            # 删除未使用的库所（无任何输入或输出弧的库所）
            unused_places = [place for place in net.places if
                             len([arc for arc in net.arcs if arc.source == place or arc.target == place]) == 0]

            for place in unused_places:
                net.places.remove(place)
            #删除弧
            t1_predecessors = get_predecessors(net, t1)
            t1_successors = get_successors(net, t1)
            t2_predecessors = get_predecessors(net, t2)
            t2_successors = get_successors(net, t2)
            for arc in list(net.arcs):
                if ((arc.source == t1 and arc.target in t2_predecessors) or
                        (arc.source in t1_successors and arc.target == t2) or
                        (arc.source == t2 and arc.target in t1_predecessors) or
                        (arc.source in t2_successors and arc.target == t1)):
                    net.arcs.remove(arc)
                else:
                    if arc.source in [t1, t2]:
                        net.arcs.remove(arc)
                        net.arcs.add(PetriNet.Arc(new_transition, arc.target))
                    if arc.target in [t1, t2]:
                        net.arcs.remove(arc)
                        net.arcs.add(PetriNet.Arc(arc.source, new_transition))


            net.transitions.remove(t1)
            net.transitions.remove(t2)

            to_remove.add(i)
            to_remove.add(j)

            transitions = [t for t in transitions if t not in [t1, t2]]
            transitions.append(new_transition)

            # 将聚合信息添加到 merged_transitions 字典
            merged_transitions[new_label] = merged_transitions.get(new_label, []) + [t1.label, t2.label]

            print(f"Aggregated {t1.label} and {t2.label} into {new_label}")

        #移除未使用的库所和边,多余的终端库所
        net = remove_unused_places_and_arcs(net, merged_transitions)
        net = remove_duplicate_arcs(net)
        net = remove_redundant_end_places(net)

        transition_labels = [t.label for t in transitions]
        similarity_matrix = calculate_semantic_relevance_batch(transition_labels)
        structure_matrix = generate_structure_matrix(net, transitions)
        #输出信息
        iteration += 1
        output_filename = f"student_petri_net_iter_{iteration}.png"
        pm4py.save_vis_petri_net(net, im, fm, output_filename)
        print(f"Saved current Petri net as {output_filename}")
        num_transitions = len(net.transitions)
        print("Number of transitions:", num_transitions)
        num_places = len(net.places)
        print("Number of places:", num_places)
        num_arcs = len(net.arcs)
        print("Number of arcs:", num_arcs)

    print("Final transitions after all possible aggregations:")
    print([t.label for t in net.transitions])

    print("Merged transitions:")
    print(merged_transitions)

    return merged_transitions



# 开始聚合
threshold = 0.7
print("Starting transition aggregation...")
aggregate_transitions(net, relevance_matrix, structure_matrix, threshold, fm)

print("Transition aggregation completed.")

# 输出聚合后的变迁
print("Final transitions:")
print([t.label for t in net.transitions])
def map_transitions(net):
    abstracted_transitions = {}

    for transition in net.transitions:
        if transition.label:
            # 使用 generate_aggregated_label 函数将变迁标签映射到抽象层次
            abstract_label = generate_aggregated_label(transition.label)
            abstracted_transitions[transition.label] = abstract_label
            # 更新变迁的标签为抽象层次标签
            transition.label = abstract_label
            print(f"Mapped transition '{transition.label}' to abstract level '{abstract_label}'")

    return abstracted_transitions
# 在变迁聚合完成后映射到抽象层次
print("Mapping transitions to abstract levels...")
abstracted_transitions = map_transitions(net)
# 输出映射后的变迁
print("Final transitions:")
print([t.label for t in net.transitions])

net=remove_duplicate_arcs(net)
net = remove_redundant_end_places(net)

# 将 Petri 网导出为 PNG 文件
pm4py.save_vis_petri_net(net, im, fm, "student_output_petri_net.png")

# 输出变迁的数目
num_transitions = len(net.transitions)
print("Number of transitions:", num_transitions)
num_places = len(net.places)
print("Number of places:", num_places)
num_arcs = len(net.arcs)
print("Number of arcs:", num_arcs)