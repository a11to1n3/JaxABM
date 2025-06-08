# PyPI Publishing Setup

This document explains how to set up automatic PyPI publishing for JaxABM using GitHub Actions.

## Overview

The automatic publishing workflow (`publish.yml`) will:

1. **Trigger on version tags** (e.g., `v0.1.1`, `v0.2.0`)
2. **Run full test suite** across Python 3.9, 3.10, 3.11
3. **Verify 70%+ test coverage** requirement
4. **Build the package** using modern Python build tools
5. **Publish to Test PyPI** first (for verification)
6. **Publish to PyPI** if tests pass
7. **Create GitHub Release** automatically

## Required Setup

### 1. PyPI API Tokens

You need to create API tokens for both PyPI and Test PyPI:

#### PyPI Token:
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Scroll to "API tokens" and click "Add API token"
3. Name: `JaxABM GitHub Actions`
4. Scope: `Entire account` (or `Specific project: jaxabm` once published)
5. Copy the generated token (starts with `pypi-`)

#### Test PyPI Token:
1. Go to [Test PyPI Account Settings](https://test.pypi.org/manage/account/)
2. Scroll to "API tokens" and click "Add API token"
3. Name: `JaxABM GitHub Actions Test`
4. Scope: `Entire account`
5. Copy the generated token (starts with `pypi-`)

### 2. GitHub Repository Secrets

Add the tokens as repository secrets:

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add:

   - **Name:** `PYPI_API_TOKEN`
   - **Value:** Your PyPI token (including `pypi-` prefix)

4. Click **New repository secret** again and add:

   - **Name:** `TEST_PYPI_API_TOKEN`
   - **Value:** Your Test PyPI token (including `pypi-` prefix)

### 3. GitHub Environments (Recommended)

For additional security, set up environments:

1. Go to **Settings** → **Environments**
2. Create environment: `pypi`
   - Add protection rule: "Required reviewers" (optional)
   - Add environment secret: `PYPI_API_TOKEN`
3. Create environment: `test-pypi`
   - Add environment secret: `TEST_PYPI_API_TOKEN`

## How to Publish a New Version

### Automatic Publishing (Recommended)

1. **Update version numbers** in:
   - `jaxabm/__init__.py`
   - `setup.py`
   - `CITATION.cff`
   - `README.md`
   - `docs/changelog.rst`

2. **Commit and push changes:**
   ```bash
   git add -A
   git commit -m "Release v0.1.2: [description]"
   git push origin main
   ```

3. **Create and push tag:**
   ```bash
   git tag -a v0.1.2 -m "Version 0.1.2: [description]"
   git push origin --tags
   ```

4. **GitHub Actions will automatically:**
   - Run all tests
   - Verify 70%+ coverage
   - Build the package
   - Publish to Test PyPI
   - Publish to PyPI
   - Create GitHub Release

### Manual Publishing (Fallback)

If automatic publishing fails, you can publish manually:

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build package
python -m build

# Check package
python -m twine check dist/*

# Upload to Test PyPI first
python -m twine upload --repository testpypi dist/*

# If test successful, upload to PyPI
python -m twine upload dist/*
```

## Workflow Features

### Quality Gates

The workflow includes several quality gates:

- ✅ **Multi-Python Testing:** Tests on Python 3.9, 3.10, 3.11
- ✅ **Coverage Verification:** Enforces 70%+ test coverage
- ✅ **Package Validation:** Validates package before publishing
- ✅ **Test PyPI First:** Tests on Test PyPI before production

### Security Features

- 🔒 **Environment Protection:** Uses GitHub environments for secrets
- 🔒 **Token Scoping:** API tokens are scoped to specific purposes
- 🔒 **Tag-Only Publishing:** Only publishes on version tags
- 🔒 **Review Gates:** Optional human approval for releases

### Monitoring

You can monitor the publishing process:

1. **GitHub Actions Tab:** Shows workflow status
2. **PyPI Project Page:** Confirms successful upload
3. **GitHub Releases:** Automatic release creation
4. **Email Notifications:** PyPI sends confirmation emails

## Troubleshooting

### Common Issues

**"Package already exists"**
- Version number may already be published
- Update version number and create new tag

**"Invalid API token"**
- Check that token is correctly copied
- Verify token has correct permissions
- Regenerate token if needed

**"Tests failing"**
- Fix failing tests before releasing
- Ensure 70%+ coverage is maintained

**"Build errors"**
- Check `setup.py` configuration
- Verify all dependencies are listed

### Support

- Check [GitHub Actions logs](https://github.com/a11to1n3/JaxABM/actions) for detailed error messages
- Review [PyPI publishing guide](https://packaging.python.org/tutorials/packaging-projects/)
- Check [Test PyPI](https://test.pypi.org/project/jaxabm/) for test uploads

## Version Naming Convention

Follow semantic versioning:

- **Patch releases** (`v0.1.1`, `v0.1.2`): Bug fixes, test improvements
- **Minor releases** (`v0.2.0`, `v0.3.0`): New features, backward compatible
- **Major releases** (`v1.0.0`, `v2.0.0`): Breaking changes

## Package Information

Once published, users can install JaxABM with:

```bash
pip install jaxabm
```

Or specific versions:

```bash
pip install jaxabm==0.1.1
```

The package will be available at:
- **PyPI:** https://pypi.org/project/jaxabm/
- **Test PyPI:** https://test.pypi.org/project/jaxabm/ 