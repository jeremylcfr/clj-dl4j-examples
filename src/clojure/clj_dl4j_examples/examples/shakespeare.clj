(ns clj-dl4j-examples.examples.shakespeare
  (:require [clj-dl4j.core :as core]
            [clj-dl4j.datasets :as datasets]
            [clj-datavec.records.csv :as csv]
            [clj-datavec.records :as records]
            [clj-nd4j.ndarray :as nda]
            [clj-nd4j.dataset.normalization :as norm]
            [clj-java-commons.core :refer [->char-array]]
            [clojure.string :as str]
            [clojure.java.io :as io])
  (:import [clj_dl4j_examples.examples.shakespeare CharacterIterator]
           [org.apache.commons.io FileUtils]
           [java.util Random]
           [java.io File IOException]
           [java.net URL]
           [java.nio.charset Charset]))

;; Adapted from original Java Code :
;; https://github.com/deeplearning4j/deeplearning4j-examples/blob/686db99fee3d4825ee70663e1a15aa8d6216f2c2/oreilly-book-dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/LSTMCharModellingExample.java
;; --------------------------------------------------------------------------------------------------
;; LSTM Character modelling example
;; @author Alex Black
;;   Example: Train a LSTM RNN to generates text, one character at a time.
;;	This example is somewhat inspired by Andrej Karpathy's blog post,
;;	"The Unreasonable Effectiveness of Recurrent Neural Networks"
;;	http://karpathy.github.io/2015/05/21/rnn-effectiveness/
;;	This example is set up to train on the Complete Works of William Shakespeare, downloaded
;;	from Project Gutenberg. Training on other text sources should be relatively easy to implement.
;;    For more details on RNNs in DL4J, see the following:
;;    http://deeplearning4j.org/usingrnns
;;    http://deeplearning4j.org/lstm
;;    http://deeplearning4j.org/recurrentnetwork

(def charsets 
  {:utf-8 "UTF-8"})

(defn charset
  ^Charset
  ([]
   (Charset/defaultCharset))
  ([type]
   (Charset/forName ^String 
     (if (string? type)
       type
       (->> (name type) (str/lower-case) (keyword) (get charsets))))))

(defn charset?
  [obj]
  (instance? Charset obj))

(defn ->charset
  ^Charset
  ([]
   (charset))
  ([obj]
   (if (charset? obj)
     charset
     (charset obj))))

(defn random
  ^Random
  ([]
   (Random.))
  ([seed]
   (Random. ^long (long seed))))

(defn random?
  [obj]
  (instance? Random obj))

(defn ->random
  ^Random
  ([]
   (random))
  ([obj]
   (if (random? obj)
     obj
     (random obj))))


;; String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength, char [] validCharacters, Random rng
(defn character-iterator
  ^CharacterIterator 
  [path encoding mini-batch-size example-length valid-characters rng]
  (CharacterIterator. 
    ^String path 
    ^Charset (->charset encoding) 
    ^int (int mini-batch-size) 
    ^int (int example-length) 
    ^chars (->char-array valid-characters) 
    ^Random (->random rng)))

(defn build-shakespeare-iterator
  ([]
   (build-shakespeare-iterator {}))
  ([{:keys [mini-batch-size sequence-length] :or {mini-batch-size 32 , sequence-length 1000}}]
   (build-shakespeare-iterator mini-batch-size sequence-length))
  ([mini-batch-size sequence-length]
   (let [input-url         "https://raw.githubusercontent.com/KonduitAI/dl4j-test-resources/master/src/main/resources/word2vec/shakespeare.txt"
         local-folder      "data"
         local-path        (str local-folder "/" "shakespeare.txt")
         local-file        (io/file local-path)
         valid-characters  (CharacterIterator/getMinimalCharacterSet)]
     (if-not (.exists ^File local-file)
       (FileUtils/copyURLToFile ^URL (URL. ^String input-url) ^String local-file)
       (println (str "Using existing text file at " (.getAbsolutePath ^File local-file))))
     (when-not (.exists ^File local-file) 
       (throw (IOException. (str "File has not been downloaded : " local-path))))
     (character-iterator local-path :utf-8 mini-batch-size sequence-length valid-characters (random 12345)))))


(defn build-network-configuration
  [^CharacterIterator shakespeare-iterator]
  (let [num-columns  (.inputColumns  shakespeare-iterator)
        num-outcomes (.totalOutcomes shakespeare-iterator)]
    {:seed 12345
     :weight-init-method :xavier
     :weights-updater {:type :rms-prop
                       :learning-rate 0.1}
     :backprop-type :truncated-bptt
     :tbptt-forward-length 50
     :tbptt-backward-length 50
     :l2-weights 0.001
     :layers [{:type           :lstm
               :activation-fn  :tanh
               :n-in           num-columns
               :n-out          200}
              {:type           :lstm
               :activation-fn  :tanh
               :n-in           200
               :n-out          200}
              {:type           :rnn-output
               :activation-fn  :softmax
               :loss-fn        :mc-xent
               :n-in           200
               :n-out          num-outcomes}]
     :training-listener {:type           :score-iteration
                         :nb-iterations  1}}))

(defn run!
  []
  (let [shakespeare-iterator  (build-shakespeare-iterator)
        network-configuration (build-network-configuration shakespeare-iterator)
        network               (core/multi-layer-network network-configuration)]
    network))

