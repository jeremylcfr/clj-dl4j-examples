(defproject io.github.jeremylcfr/clj-dl4j-examples "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.clojure/algo.generic "0.1.3"]
                 [jfree/jcommon "1.0.16"]
                 [org.jfree/jfreechart "1.5.4"]
                 [commons-io/commons-io "2.11.0"]
                 ;; [org.nd4j/nd4j-cuda-11.6-platform "1.0.0-M2.1"]
                 [io.github.jeremylcfr/clj-java-commons "1.1.0-SNAPSHOT"]
                 [io.github.jeremylcfr/clj-nd4j "0.1.0-SNAPSHOT"]
                 [io.github.jeremylcfr/clj-datavec "0.1.0-SNAPSHOT"]
                 [io.github.jeremylcfr/clj-dl4j "0.1.0-SNAPSHOT"]]
  :source-paths ["src/clojure"]
  :java-source-paths ["src/java"]
  :repl-options {:init-ns clj-dl4j-examples.core})
