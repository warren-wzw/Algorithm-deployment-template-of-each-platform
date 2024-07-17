import os
import sys
os.chdir(sys.path[0])
import onnx
from onnx import helper,shape_inference
ONNXMODEL='yolov5n.onnx'

def main():
    model = onnx.load(ONNXMODEL)
    graph = model.graph
    node  = graph.node
    """"1"""
    node_name1='/model.11/Slice'
    node_name2='/model.11/Concat_1'
    #创建 Reshape
    new_shape = [2, 1, 1, 1, 1]
    value=helper.make_tensor('reshape_value', 7, dims=[len(new_shape)], vals=new_shape)
    attra=helper.make_attribute("value",value)
    shape_node=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_shape'],
        name='reshape_shape'
    )
    shape_node.attribute.insert(0,attra)
    reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.11/Slice_output_0','reshape_shape'],
        outputs=["/model.11/Slice_reshaped"],
        name='/model.11/Slice_reshaped'
    )
    #insert node
    for i,node_ in enumerate(node):
        if node_.name=='/model.11/Slice':
            node.insert(i+1,shape_node)
            node.insert(i+2,reshape_node)
    """create new concat node"""
    Concat_node=onnx.helper.make_node(
        'Concat',
        inputs=['/model.11/Slice_reshaped','/model.11/Cast_output_0'],
        outputs=['/model.11/Concat_1_output_0'],
        name='/model.11/Concat_1',
        axis=0  
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name=='/model.11/Concat_1':
            graph.node.remove(node_)
            graph.node.insert(i, Concat_node)
            print("-------------------------------")
    """"2"""
    node_name3='/model.15/Slice'
    node_name4='/model.15/Concat_1'

    # # 创建 Reshape
    tensor_values = [2, 1, 1, 1, 1]  
    value2=helper.make_tensor('reshape_value', 7, dims=[len(tensor_values)], vals=tensor_values)
    attra2=helper.make_attribute("value",value2)
    shape_node2=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_shape2'],
        name='reshape_shape2'
    )
    shape_node2.attribute.insert(0,attra2)
    reshape_node2 = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.15/Slice_output_0','reshape_shape2'],
        outputs=["/model.15/Slice_reshaped"],
        name='/model.15/Slice_reshaped'
    )
    #insert node
    for i,node_ in enumerate(node):
        if node_.name==node_name3:
            node.insert(i+1,shape_node2)
            node.insert(i+2,reshape_node2)
    """create new concat node"""
    Concat_node2=onnx.helper.make_node(
        'Concat',
        inputs=['/model.15/Slice_reshaped','/model.15/Cast_output_0'],
        outputs=['/model.15/Concat_1_output_0'],
        name='/model.15/Concat_1',
        axis=0  
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name4:
            #print(node)
            graph.node.remove(node_)
            graph.node.insert(i, Concat_node2)
            print("-------------------------------")

    """3"""
    node_name_24_reshape='/model.24/Reshape'
    node_name_24_transpose='/model.24/Transpose'
    node_name_24_costant='/model.24/Constant'
    """create reshape"""
    reshape_24_shape1 = [1, 255, 80, 80]
    value=helper.make_tensor('reshape_value', 7, dims=[len(reshape_24_shape1)], vals=reshape_24_shape1)
    reshape_24_attr1=helper.make_attribute("value",value)
    constant_24_1=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_24_1'],
        name='reshape_24_1'
    )
    constant_24_1.attribute.insert(0,reshape_24_attr1)
    reshape_node3 = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.24/m.0/Conv_output_0','reshape_24_1'],
        outputs=["/model.24/Reshape_output_0"],
        name='/model.24/Reshape_output_0'
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_reshape :
            graph.node.remove(node_)
            node.insert(i,reshape_node3)
            print("-------------------------------")
            break
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_costant :
            graph.node.remove(node_)
            node.insert(i,constant_24_1)
            print("-------------------------------")
            break
    """create Transpose"""
    transposenode_24_1 = onnx.helper.make_node(
    'Transpose',
    inputs=['/model.24/Reshape_output_0'],
    outputs=['/model.24/Transpose_output_0_'],
    name='/model.24/Transpose_new1',
    )
    newattr = helper.make_attribute("perm", [0,2,3,1])
    transposenode_24_1.attribute.insert(0,newattr)
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_transpose:
            graph.node.remove(node_)
            node.insert(i,transposenode_24_1)
            print("-------------------------------")
            break
    """create reshape"""
    nodename_transpose_24_1='/model.24/Transpose_new1'
    reshape_24_shape2 = [1, 3, 80, 80, 85]
    value=helper.make_tensor('reshape_value', 7, dims=[len(reshape_24_shape2)], vals=reshape_24_shape2)
    reshape_24_attr2=helper.make_attribute("value",value)
    constant_24_2=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_shape4'],
        name='reshape_shape4'
    )
    constant_24_2.attribute.insert(0,reshape_24_attr2)
    reshape_24_node2 = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.24/Transpose_output_0_','reshape_shape4'],
        outputs=["/model.24/Transpose_output_0"],
        name='/model.24/Transpose_output_0_reshaped'
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==nodename_transpose_24_1 :
            node.insert(i+1,constant_24_2)
            node.insert(i+2,reshape_24_node2)
            print("-------------------------------")
            break
        
    """4"""
    node_name_24_reshape2='/model.24/Reshape_2'
    node_name_24_transpose2='/model.24/Transpose_1'
    node_name_24_costant2='/model.24/Constant_8'
    """create reshape"""
    reshape_24_shape3 = [1, 255, 40, 40]
    value=helper.make_tensor('reshape_value', 7, dims=[len(reshape_24_shape3)], vals=reshape_24_shape3)
    reshape_24_attr2=helper.make_attribute("value",value)
    constant_24_3=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_24_2'],
        name='reshape_24_2'
    )
    constant_24_3.attribute.insert(0,reshape_24_attr2)
    reshape_node3 = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.24/m.1/Conv_output_0','reshape_24_2'],
        outputs=["/model.24/Reshape_2_output_0"],
        name='/model.24/Reshape_2_output_0'
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_reshape2 :
            graph.node.remove(node_)
            node.insert(i,reshape_node3)
            print("-------------------------------")
            break
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_costant2 :
            graph.node.remove(node_)
            node.insert(i,constant_24_3)
            print("-------------------------------")
            break
    """create Transpose"""
    transposenode_24_2 = onnx.helper.make_node(
    'Transpose',
    inputs=['/model.24/Reshape_2_output_0'],
    outputs=['/model.24/Transpose_1_output_0_'],
    name='/model.24/Transpose_new2',
    )
    newattr = helper.make_attribute("perm", [0,2,3,1])
    transposenode_24_2.attribute.insert(0,newattr)
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_transpose2:
            graph.node.remove(node_)
            node.insert(i,transposenode_24_2)
            print("-------------------------------")
            break
    """create reshape"""
    nodename_transpose_24_2='/model.24/Transpose_new2'
    reshape_24_shape4 = [1, 3, 40, 40, 85]
    value=helper.make_tensor('reshape_value', 7, dims=[len(reshape_24_shape4)], vals=reshape_24_shape4)
    reshape_24_attr2=helper.make_attribute("value",value)
    constant_24_4=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_shape5'],
        name='reshape_shape5'
    )
    constant_24_4.attribute.insert(0,reshape_24_attr2)
    reshape_24_node2 = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.24/Transpose_1_output_0_','reshape_shape5'],
        outputs=["/model.24/Transpose_1_output_0"],
        name='/model.24/Transpose_output_2_reshaped'
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==nodename_transpose_24_2 :
            node.insert(i+1,constant_24_4)
            node.insert(i+2,reshape_24_node2)
            print("-------------------------------")
            break
    """5"""
    node_name_24_reshape3='/model.24/Reshape_4'
    node_name_24_transpose3='/model.24/Transpose_2'
    node_name_24_costant3='/model.24/Constant_16'
    """create reshape"""
    reshape_24_shape5 = [1, 255, 20, 20]
    value=helper.make_tensor('reshape_value', 7, dims=[len(reshape_24_shape5)], vals=reshape_24_shape5)
    reshape_24_attr4=helper.make_attribute("value",value)
    constant_24_5=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_24_6'],
        name='reshape_24_6'
    )
    constant_24_5.attribute.insert(0,reshape_24_attr4)
    reshape_node5 = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.24/m.2/Conv_output_0','reshape_24_6'],
        outputs=["/model.24/Reshape_4_output_0"],
        name='/model.24/Reshape_4_output_0'
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_reshape3 :
            graph.node.remove(node_)
            node.insert(i,reshape_node5)
            print("-------------------------------")
            break
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_costant3 :
            graph.node.remove(node_)
            node.insert(i,constant_24_5)
            print("-------------------------------")
            break
    """create Transpose"""
    transposenode_24_3 = onnx.helper.make_node(
    'Transpose',
    inputs=['/model.24/Reshape_4_output_0'],
    outputs=['/model.24/Transpose_2_output_0_'],
    name='/model.24/Transpose_new3',
    )
    newattr = helper.make_attribute("perm", [0,2,3,1])
    transposenode_24_3.attribute.insert(0,newattr)
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==node_name_24_transpose3:
            graph.node.remove(node_)
            node.insert(i,transposenode_24_3)
            print("-------------------------------")
            break
    """create reshape"""
    nodename_transpose_24_3='/model.24/Transpose_new3'
    reshape_24_shape5 = [1, 3, 20, 20, 85]
    value=helper.make_tensor('reshape_value', 7, dims=[len(reshape_24_shape5)], vals=reshape_24_shape5)
    reshape_24_attr5=helper.make_attribute("value",value)
    constant_24_6=helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_shape7'],
        name='reshape_shape7'
    )
    constant_24_6.attribute.insert(0,reshape_24_attr5)
    reshape_24_node6 = onnx.helper.make_node(
        "Reshape",
        inputs=['/model.24/Transpose_2_output_0_','reshape_shape7'],
        outputs=["/model.24/Transpose_2_output_0"],
        name='/model.24/Transpose_output_3_reshaped'
    )
    for i in range(len(graph.node)):
        node_ = graph.node[i]
        if node_.name==nodename_transpose_24_3 :
            node.insert(i+1,constant_24_6)
            node.insert(i+2,reshape_24_node6)
            print("-------------------------------")
            break


    """check model"""
    onnx.checker.check_model(model)
    inferred_model = shape_inference.infer_shapes(model)
    # 保存修改后的模型
    onnx.save(model, "changed_model.onnx")

if __name__=="__main__":
    main()