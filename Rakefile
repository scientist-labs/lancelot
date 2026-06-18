# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/extensiontask"

# Dev-only tooling (rspec, standard) is absent from the precompiled-gem build
# environment (the cross-gem container and the `gem build` step install only the
# runtime deps). Guard each require so loading the Rakefile to drive `native:<plat>`
# / `compile` never explodes on a missing development gem. The tasks they define are
# only registered when the gem is present.
begin
  require "rspec/core/rake_task"
  RSpec::Core::RakeTask.new(:spec)
rescue LoadError
  # rspec unavailable (e.g. precompiled-gem build) -> skip the :spec task.
end

begin
  require "standard/rake"
rescue LoadError
  # standard unavailable -> skip its rake tasks.
end

task build: :compile

# Load the gemspec ONCE and hand it to Rake::ExtensionTask so that, in addition to
# the plain `compile` task, rake-compiler generates the per-platform `native:<plat>`
# tasks the precompiled-gem release pipeline drives (e.g. native:x86_64-linux).
GEMSPEC = Gem::Specification.load("lancelot.gemspec")

Rake::ExtensionTask.new("lancelot", GEMSPEC) do |ext|
  ext.lib_dir = "lib/lancelot"
  # Enable cross-compilation so `rake native:<platform>` exists for every platform
  # the release matrix builds. The linux legs run inside rb-sys-dock; arm64-darwin
  # builds natively on an Apple-Silicon runner.
  ext.cross_compile = true
  ext.cross_platform = %w[
    x86_64-linux
    aarch64-linux
    arm64-darwin
  ]
end

# Default task: only depend on tasks that always exist. `spec`/`standard` are
# conditionally defined above, so reference them lazily to avoid a hard failure
# when the dev gems are absent.
desc "Compile, then run specs and the linter when available"
task :default do
  Rake::Task["clobber"].invoke
  Rake::Task["compile"].invoke
  Rake::Task["spec"].invoke if Rake::Task.task_defined?("spec")
  Rake::Task["standard"].invoke if Rake::Task.task_defined?("standard")
end
