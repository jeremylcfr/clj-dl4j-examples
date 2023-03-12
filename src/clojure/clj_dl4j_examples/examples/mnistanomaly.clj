(ns clj-dl4j-examples.examples.mnistanomaly
  (:require [clojure.algo.generic.functor :refer [fmap]]
            [clj-nd4j.ndarray :as nd4j]
            [clj-nd4j.dataset :as dt]
            [clj-dl4j.core :as core]
            [clj-dl4j.resources.datasets :as dts])
  (:import [clj_dl4j_examples.examples.mnistanomaly MNISTVisualizer]

           ;; Wrapped/hinted
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork])
  (:refer-clojure :exclude [run!]))

(def data (volatile! nil))

(def mnistanomaly-network-configuration
  {:seed 12345
   :weight-init-method :xavier
   :weights-updater {:type :adagrad
                     :learning-rate 0.05}
   :activation-fn :relu
   :l2-weights 0.0001
   :layers [{:type :dense
             :n-in 784
             :n-out 250}
            {:type :dense
             :n-in 250
             :n-out 10}
            {:type :dense
             :n-in 10
             :n-out 250}
            {:type                   :output
             :n-in                   250
             :n-out                  784
             :loss-fn                :mse}]
   :training-listener {:type         :score-iteration
                       :nb-iterations  10}})

(defn build-network
  []
  (core/multi-layer-network mnistanomaly-network-configuration))

(defn ->dataset-iterator
  []
  (iterator-seq
    (dts/mnist-dataset-iterator 100 50000 false)))

(defn prepare-data
  []
  (let [iterator (->dataset-iterator)
        random (java.util.Random. 12345)]
    (reduce
      (fn [agg dataset]
        (let [{:keys [train test]} (dt/deep-split-test-and-train dataset 80 random)
              train-features (:features train)
              test-features (:features test)
              test-labels (nd4j/arg-max (:labels test) [1])]
          (-> (update agg :train-features conj train-features)
              (update     :test-features  conj test-features)
              (update     :test-labels    conj test-labels))))
      {:train-features []
       :test-features  []
       :test-labels    []} iterator)))

(defn prepare-data!
  []
  (vreset! data (prepare-data)))

(defn train!
  [^MultiLayerNetwork network nb-trains]
  (doseq [epoch (range nb-trains)]
    (doseq [feature (:train-features @data)]
      (.fit network feature feature))
    (println (str "Epoch " epoch " complete"))))

(defn fit-and-score-by-digits
  [^MultiLayerNetwork network]
  (let [{:keys [test-features test-labels]} @data]
    (reduce-kv
      (fn [super-agg idx feature]
        (let [label (nth test-labels idx)
              nrows (.rows feature)]
          (reduce
            (fn [agg i]
              (let [example (.getRow feature i true)
                    digit (long (.getDouble label i))
                    score (.score network (dt/->dataset example example))]
                (update agg digit conj {:score score :example example})))
            super-agg (range nrows))))
      {} test-features)))

(defn sort-results
  [results]
  (reduce-kv
    (fn [agg digit shard]
      (let [sorted (->> (sort-by :score shard)
                        (map :example))
            rsorted (reverse sorted)]
        (-> (update agg :best  concat (take 5 sorted))
            (update     :worst concat (take 5 rsorted)))))
    {} results))


;; ! Use this !
;; BEWARE : training on 30 periods is a pretty long task (and a pretty overkill one too)
(defn run!
  ([]
   (run! 30))
  ([nb-trains]
   (let [network (build-network)]
     (prepare-data!)
     (train! network nb-trains)
     (let [fitted (fit-and-score-by-digits network)
           {:keys [best worst]} (sort-results fitted)]
       (.visualize (MNISTVisualizer. 2.0 best "Best (Low Rec. Error)"))
       (.visualize (MNISTVisualizer. 2.0 worst "Worst (High Rec. Error)"))))))
