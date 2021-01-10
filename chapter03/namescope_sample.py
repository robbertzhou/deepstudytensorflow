import tensorflow as tf

def namespace_scope_in_graph():
    g1 = tf.Graph()
    with g1.as_default():
        with g1.name_scope("layer1"):
            c1 = tf.constant(250,name="c1")

    return g1,c1

if __name__ == "__main__":
    g1 , c1 = namespace_scope_in_graph()
    print("c1的张量：{}".format(c1))
    print("c1的名称：{}".format(c1.name))