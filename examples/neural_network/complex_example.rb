# Author::    Masatoshi SEKI
# License::   MPL 1.1
# Project::   ai4r
# Url::       http://ai4r.rubyforge.org/
#
# You can redistribute it and/or modify it under the terms of 
# the Mozilla Public License version 1.1  as published by the 
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt

require File.dirname(__FILE__) + '/../../lib/ai4r/neural_network/complex_backpropagation'
require 'benchmark'

pattern_t = [Complex(-0.6, 0.6),
             Complex(-0.3, 0.6),
             Complex(0.0, 0.6),
             Complex(0.3, 0.6),
             Complex(0.6, 0.6),
             Complex(0.0, 0.3),
             Complex(0.0, 0.0),
             Complex(0.0, -0.3),
             Complex(0.0, 0.3)]

pattern_circle = (0..11).collect {|n|
  t = n * Math::PI / 6
  Complex(Math.cos(t), Math.sin(t))
}

def to_nn(complex)
  (complex + Complex(1,1)) * 0.5
end

def from_nn(c)
  c * 2.0 - Complex(1,1)
end

def rotate(pattern)
  rot = Complex(Math.cos(0.25 * Math::PI), Math.sin(0.25 * Math::PI))
  pattern.each do |pt1|
    yield(to_nn(pt1), to_nn(pt1 * rot))
  end
end

def print_pt(pt1, pt2)
  printf("(%.3f %.3f) (%.3f %.3f)\n", pt1.real, pt1.image, pt2.real, pt2.image)
end

times = Benchmark.measure do
  srand 1
  net = Ai4r::NeuralNetwork::Backpropagation.new([2, 12, 2])
  
  puts "Training the network, please wait."
    201.times do |i|
    error = 0.0
    rotate(pattern_t) do |pt1, pt2|
      error = net.train([pt1.real, pt1.image], [pt2.real, pt2.image])
    end
    puts "Error after iteration #{i}:\t#{error}" if i%20 == 0
  end

  pattern_circle.each do |pt|
    pt2 = to_nn(pt)
    x, y = net.eval([pt2.real, pt2.image])
    print_pt(pt, from_nn(Complex(x, y)))
  end
end

puts "Elapsed time: #{times}"

times = Benchmark.measure do
  srand 1
  net = Ai4r::NeuralNetwork::ComplexBackpropagation.new([1, 6, 1])
  
  rotate(pattern_t) do |pt1, pt2|
    print_pt(pt1, pt2)
    error = net.train([pt1], [pt2])
  end

  puts "Training the network, please wait."
    201.times do |i|
    error = 0.0
    rotate(pattern_t) do |pt1, pt2|
      error = net.train([pt1], [pt2])
    end
    puts "Error after iteration #{i}:\t#{error}" if i%20 == 0
  end

  pattern_circle.each do |pt|
    print_pt(pt, from_nn(net.eval([pt])[0]))
  end
end

puts "Elapsed time: #{times}"
