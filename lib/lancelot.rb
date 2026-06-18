# frozen_string_literal: true

require_relative "lancelot/version"

# Load the compiled Rust extension. Precompiled (platform) gems install it into a
# Ruby-ABI-versioned subdir (lib/lancelot/<major.minor>/lancelot.{so,bundle}) so a
# single fat gem can carry a binary per Ruby version; source/dev builds place it flat
# at lib/lancelot/lancelot.{so,bundle}. Try the versioned path first, fall back to the
# flat one. Resolution goes through $LOAD_PATH (`require`, never `require_relative`)
# because RubyGems installs native extensions outside the gem's lib/ dir.
begin
  RUBY_VERSION =~ /(\d+\.\d+)/
  require "lancelot/#{Regexp.last_match(1)}/lancelot"
rescue LoadError
  require "lancelot/lancelot"
end

require_relative "lancelot/dataset"
require_relative "lancelot/rank_fusion"

module Lancelot
  class Error < StandardError; end
end
