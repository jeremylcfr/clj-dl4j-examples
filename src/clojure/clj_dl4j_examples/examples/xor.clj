(ns clj-dl4j-examples.examples.xor
  (:require [clj-dl4j.core :as core]
            [clj-nd4j.dataset :as dataset]
            [clj-nd4j.ndarray :as nd4j]
            [clj-dl4j.datasets :as ddataset]
            [clj-java-commons.coerce :refer [->clj]])
  (:import [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]))

(def xor-training-data
  (dataset/->dataset
    ;; Features [x y]
    [[0 0] [0 1] [1 0] [1 1]]
    ;; Labels (xor x y)
    ;; with [0 1] = false and [1 0] = true
    [[0 1] [1 0] [1 0] [0 1]]))

;; Datavec version from file
;; Vectorize data as behind
;; (def xor-training-data
;;   (ddataset/read-datasets :jackson
;;     {:type-hint      :regular
;;      :dataset-type   :classification
;;      :use-nd4j?      false
;;      :schema         [:x :y :xor]
;;      :batch-size     4
;;      :label-idx      2
;;      :max-labels     2}
;;     "data/xor.json"))

(def xor-network-configuration
  {:mini-batch false
   :seed 123
   :weights-updater {:type :sgd
                     :learning-rate 0.1}
   :bias-init 0
   :layers [{:type                   :dense
             :n-in                   2
             :n-out                  4
             :activation-fn          :sigmoid
             :weight-distribution    {:type   :uniform
                                      :lower  0
                                      :upper  1}}
            {:type                   :output
             :n-in                   4
             :n-out                  2
             :activation-fn          :softmax
             :loss-fn                :negative-log-likehood
             :weight-distribution    {:type   :uniform
                                      :lower  0
                                      :upper  1}}]
   :training-listener {:type         :score-iteration
                       :nb-iterations  100}})

(defn build-network
  ^MultiLayerNetwork
  []
  (core/multi-layer-network xor-network-configuration))

(def network (volatile! nil))

(defn build!
  []
  (vreset! network (build-network)))

(defn train!
  ([]
   (train! 1000))
  ([n]
   (if-let [network* @network]
     (dotimes [_ n] (.fit ^MultiLayerNetwork network* ^DataSet xor-training-data))
     (do (build!) (train! n)))))

(defn translate!
  [^INDArray a]
  (let [result (->clj a)]
    (println (str "Score for XOR({FALSE ; FALSE}) ==================> " (/ (Math/round ^double (* 100000 (get-in result [0 0]))) 1000.0) "% TRUE"))
    (println (str "Score for XOR({ TRUE ; FALSE}) ==================> " (/ (Math/round ^double (* 100000 (get-in result [1 0]))) 1000.0) "% TRUE"))
    (println (str "Score for XOR({FALSE ;  TRUE}) ==================> " (/ (Math/round ^double (* 100000 (get-in result [2 0]))) 1000.0) "% TRUE"))
    (println (str "Score for XOR({ TRUE ;  TRUE}) ==================> " (/ (Math/round ^double (* 100000 (get-in result [3 0]))) 1000.0) "% TRUE"))))

(defn show!
  ([]
   (show! (.getFeatures ^DataSet xor-training-data)))
  ([data]
    (if-let [network* @network]
     (translate! (.output ^MultiLayerNetwork network* ^INDArray (nd4j/nd-array :matrix data)))
     (do (build!) (show! data)))))
