# Author::    Masatoshi SEKI
# License::   MPL 1.1
# Project::   ai4r
# Url::       http://ai4r.rubyforge.org/
#
# You can redistribute it and/or modify it under the terms of 
# the Mozilla Public License version 1.1  as published by the 
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt

require File.dirname(__FILE__) + '/backpropagation'
require 'complex'

module Ai4r
  module NeuralNetwork
    class ComplexBackpropagation < Backpropagation
      def initialize(network_structure)
        super
        @initial_weight_function = lambda { |n, i, j|
          Complex(((rand 2000)/1000.0) - 1, ((rand 2000)/1000.0) - 1)
        }
        @propagation_function = lambda { |x|
          Complex(1/(1+Math.exp(-1*(x.real))), 1/(1+Math.exp(-1*(x.image))))
        }
        @derivative_propagation_function = lambda { |y, e| 
          Complex(e.real * y.real*(1-y.real),
                  e.image * y.image*(1-y.image))
        }
      end

      def make_zero
        Complex(0.0, 0.0)
      end

      def make_one
        Complex(1.0, 1.0)
      end
      protected

      def calculate_change(n, i, j)
        @deltas[n][j]*@activation_nodes[n][i].conjugate
      end

      # Calculate quadratic error for a expected output value 
      # Error = 0.5 * sum( (expected_value[i] - output_value[i])**2 )
      def calculate_error(expected_output)
        super(expected_output).abs
      end
    end
  end
end
