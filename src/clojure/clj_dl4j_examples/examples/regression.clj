(ns clj-dl4j-examples.examples.regression
  (:require [clj-dl4j.core :as core]
            [clj-dl4j.datasets :as datasets]
            [clj-datavec.records.csv :as csv]
            [clj-datavec.records :as records]
            [clj-nd4j.ndarray :as nda]
            [clj-nd4j.dataset.normalization :as norm])

  (:import [org.deeplearning4j.eval RegressionEvaluation]

           [org.nd4j.linalg.api.ndarray INDArray]


           [org.jfree.chart ChartFactory ChartPanel JFreeChart]
           [org.jfree.chart.axis NumberAxis]
           [org.jfree.chart.plot XYPlot PlotOrientation]
           [org.jfree.data.xy XYSeries XYSeriesCollection]
           [org.jfree.ui RefineryUtilities])
  )

;; {:keys [mini-batch-size nb-possible-labels label-idx regression] :or {regression false} :as options}

;; https://github.com/deeplearning4j/deeplearning4j-examples/blob/master/oreilly-book-dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/regression/SingleTimestepRegressionExample.java

(def dataset-options
  {:input-type :csv
   :to-skip 0
   :sep ";"
   :path-template {:train "data/regression/passengers_train_%d.csv"
                   :test  "data/regression/passengers_test_%d.csv"}
   :mini-batch-size 32
   :nb-possible-labels -1
   :label-idx 1
   :regression true
   :split {:type :numbered-file
           :min-idx 0
           :max-idx 0}})


;; num-to-skip
(defn build-dataset-iterator
  [type-fn]
  (datasets/->sequence-record-reader-dataset-iterator  (update dataset-options :path-template type-fn)))

(defn build-dataset
  [type-fn]
  (.next (build-dataset-iterator type-fn)))

;; maybe one map
(def normalizer-options
  {:type :min-max-scaler
   :spec {:min-range 0.0
          :max-range 1.0}
   :options {:fit-label? true}})

(defn build-normalizer
  []
  (let [{:keys [type spec options]} normalizer-options]
    (norm/normalizer-min-max-scaler spec options)))

(def network-configuration
  {:seed 140
   :weight-init-method :xavier
   :weights-updater {:type :nesterovs
                     :learning-rate 0.0015
                     :momentum 0.9}
   :layers [{:type           :graves-lstm
             :activation-fn  :tanh
             :n-in           1
             :n-out          10}
            {:type           :rnn-output
             :activation-fn  :identity
             :loss-fn        :mse
             :n-in           10
             :n-out          1}]
   :training-listener {:type           :score-iteration
                       :nb-iterations  20}})

(defn create-plot-series
  ^XYSeriesCollection
  ([features offset plot-name]
   (create-plot-series features offset plot-name (XYSeriesCollection.)))
  ([^INDArray features offset ^String plot-name ^XYSeriesCollection series-collection]
   (let [nb-rows (aget (nda/get-shape features) 2)
         series (XYSeries. plot-name)]
     (doseq [i (range nb-rows)]
       (.add ^XYSeries series ^int (int (+ offset i))))
     series-collection)))


(defn run!
  []
  (let [train-dataset (build-dataset :train)
        test-dataset  (build-dataset :test)

        normalizer (-> (build-normalizer)
                       (norm/fit-dataset train-dataset)
                       )
        _ (norm/transform! normalizer train-dataset)
        _ (norm/transform! normalizer test-dataset)

        ;; replace
        test-features (.getFeatures test-dataset)
        test-labels (.getLabels test-dataset)

        network (core/multi-layer-network network-configuration)]
    (dotimes [k 300]
      (.fit network train-dataset)
      (let [evaluation (RegressionEvaluation. 1)
            predicted (.output network test-features, false)]
        (.evalTimeSeries evaluation test-labels predicted)
        (println (.stats evaluation))))))












;; (def data (volatile! nil))

;; (def mnistanomaly-network-configuration
;;   {:seed 12345
;;    :weight-init-method :xavier
;;    :weights-updater {:type :adagrad
;;                      :learning-rate 0.05}
;;    :activation-fn :relu
;;    :l2-weights 0.0001
;;    :layers [{:type :dense
;;              :n-in 784
;;              :n-out 250}
;;             {:type :dense
;;              :n-in 250
;;              :n-out 10}
;;             {:type :dense
;;              :n-in 10
;;              :n-out 250}
;;             {:type                   :output
;;              :n-in                   250
;;              :n-out                  784
;;              :loss-fn                :mse}]
;;    :training-listener {:type         :score-iteration
;;                        :nb-iterations  10}})

;; (defn build-network
;;   []
;;   (core/multi-layer-network mnistanomaly-network-configuration))

;; (defn ->dataset-iterator
;;   []
;;   (iterator-seq
;;    (dts/mnist-dataset-iterator 100 50000 false)))

;; (defn prepare-data
;;   []
;;   (let [iterator (->dataset-iterator)
;;         random (java.util.Random. 12345)]
;;     (reduce
;;      (fn [agg dataset]
;;        (let [{:keys [train test]} (dt/deep-split-test-and-train dataset 80 random)
;;              train-features (:features train)
;;              test-features (:features test)
;;              test-labels (nd4j/arg-max (:labels test) [1])]
;;          (-> (update agg :train-features conj train-features)
;;              (update     :test-features  conj test-features)
;;              (update     :test-labels    conj test-labels))))
;;      {:train-features []
;;       :test-features  []
;;       :test-labels    []} iterator)))

;; (defn prepare-data!
;;   []
;;   (vreset! data (prepare-data)))

;; (defn train!
;;   [^MultiLayerNetwork network nb-trains]
;;   (doseq [epoch (range nb-trains)]
;;     (doseq [feature (:train-features @data)]
;;       (.fit network feature feature))
;;     (println (str "Epoch " epoch " complete"))))

;; (defn fit-and-score-by-digits
;;   [^MultiLayerNetwork network]
;;   (let [{:keys [test-features test-labels]} @data]
;;     (reduce-kv
;;      (fn [super-agg idx feature]
;;        (let [label (nth test-labels idx)
;;              nrows (.rows feature)]
;;          (reduce
;;           (fn [agg i]
;;             (let [example (.getRow feature i true)
;;                   digit (long (.getDouble label i))
;;                   score (.score network (dt/->dataset example example))]
;;               (update agg digit conj {:score score :example example})))
;;           super-agg (range nrows))))
;;      {} test-features)))

;; (defn sort-results
;;   [results]
;;   (reduce-kv
;;    (fn [agg digit shard]
;;      (let [sorted (->> (sort-by :score shard)
;;                        (map :example))
;;            rsorted (reverse sorted)]
;;        (-> (update agg :best  concat (take 5 sorted))
;;            (update     :worst concat (take 5 rsorted)))))
;;    {} results))


;; ;; ! Use this !
;; ;; BEWARE : training on 30 periods is a pretty long task (and a pretty overkill one too)
;; (defn run!
;;   ([]
;;    (run! 30))
;;   ([nb-trains]
;;    (let [network (build-network)]
;;      (prepare-data!)
;;      (train! network nb-trains)
;;      (let [fitted (fit-and-score-by-digits network)
;;            {:keys [best worst]} (sort-results fitted)]
;;        (.visualize (MNISTVisualizer. 2.0 best "Best (Low Rec. Error)"))
;;        (.visualize (MNISTVisualizer. 2.0 worst "Worst (High Rec. Error)"))))))
