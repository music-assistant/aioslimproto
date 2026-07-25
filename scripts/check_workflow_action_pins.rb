#!/usr/bin/env ruby
# frozen_string_literal: true

require "pathname"
require "yaml"

ROOT = Pathname.new(__dir__).join("..").expand_path
WORKFLOW_GLOBS = [
  ".github/**/*.yml",
  ".github/**/*.yaml",
].freeze
PINNED_ACTION = /\A[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\/.+)?@[0-9a-f]{40}\z/
VERSION_COMMENT = /\Av?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\z/

errors = []

WORKFLOW_GLOBS.flat_map { |glob| Dir.glob(ROOT.join(glob).to_s) }.sort.uniq.each do |file|
  begin
    YAML.load_file(file)
  rescue Psych::SyntaxError => e
    errors << "#{file}: invalid YAML (#{e.message.lines.first.strip})"
    next
  end

  File.foreach(file).with_index(1) do |line, line_no|
    next unless (match = line.match(/^\s*uses:\s*(.+)$/))

    raw_value, raw_comment = match[1].split("#", 2)
    uses = raw_value.strip
    comment = raw_comment&.strip

    next if uses.start_with?("./", "../")
    next if uses.start_with?("docker://")

    unless uses.match?(PINNED_ACTION)
      errors << "#{file}:#{line_no} mutable external action ref: #{uses}"
      next
    end

    if comment.nil? || comment.empty?
      errors << "#{file}:#{line_no} pinned action is missing an exact version comment"
    elsif !comment.match?(VERSION_COMMENT)
      errors << "#{file}:#{line_no} version comment must look like an exact semver tag (got #{comment.inspect})"
    end
  end
end

if errors.empty?
  puts "No mutable external action refs or YAML errors found."
  exit 0
end

warn errors.join("\n")
exit 1
