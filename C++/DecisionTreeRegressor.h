#ifndef DECISIONTREEREGRESSOR_H
#define DECISIONTREEREGRESSOR_H

#include "DecisionTree.h"

template <typename T>
struct Node {
    bool isLeaf;
    T prediction;
    size_t featureIndex;
    T splitValue;
    Node* left;
    Node* right;

    Node() : isLeaf(true), prediction(0), featureIndex(0), splitValue(0), left(nullptr), right(nullptr) {}
};

template <typename T>
class DecisionTreeRegressor : public DecisionTree<T> {
public:
    DecisionTreeRegressor();
    ~DecisionTreeRegressor();

    void fit(int max_depth = 5, int min_samples_split = 2) override;
    std::vector<T> predict(const std::vector<std::vector<T>>& data) override;

private:
    Node<T>* treeRoot;
    T predictSingle(Node<T>* node, const std::vector<T>& x);
};

#endif
