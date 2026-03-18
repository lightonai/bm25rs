.PHONY: bump-version release test build

# Usage:
#   make bump-version v=0.2.0
#   make release v=0.2.0

bump-version:
ifndef v
	$(error Usage: make bump-version v=X.Y.Z)
endif
	@echo "Bumping version to $(v)..."
	sed -i '' 's/^version = ".*"/version = "$(v)"/' Cargo.toml
	sed -i '' 's/^version = ".*"/version = "$(v)"/' python/Cargo.toml
	sed -i '' 's/^version = ".*"/version = "$(v)"/' python/pyproject.toml
	@echo "Updated:"
	@grep '^version' Cargo.toml
	@grep '^version' python/Cargo.toml
	@grep '^version' python/pyproject.toml

release: bump-version
	git add Cargo.toml python/Cargo.toml python/pyproject.toml
	git commit -m "Release v$(v)"
	git tag v$(v)
	git push && git push --tags
	@echo "Released v$(v) — CI will publish to crates.io and PyPI"

test:
	cargo test --lib

test-cuda:
	cargo test --lib --features cuda

build:
	cd python && maturin develop --release

build-gpu:
	cd python && maturin develop --release --features cuda
