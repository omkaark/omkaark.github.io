# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a static site generator for a personal blog built in Python. The project converts Markdown blog posts into static HTML pages using a custom build system.

## Architecture

- **Build System**: `src/build.py` is the main build script that orchestrates the entire static site generation
- **Content Structure**: 
  - `src/posts.json` defines blog posts metadata (title, date)
  - `src/posts/N.md` contains markdown content for post N
  - `src/index.html` serves as the homepage template
- **Output**: Generated site is built to `docs/` directory (GitHub Pages compatible)
- **Styling**: CSS is embedded in `index.html` between `/* STYLE */` markers and extracted during build

## Key Build Process

1. **CSS Extraction**: The build script extracts base CSS from `index.html` and adds post-specific styles
2. **Index Generation**: Creates homepage by replacing `<!-- POSTS -->` placeholder with post listings
3. **Post Processing**: Converts markdown files to HTML using mistune library and generates individual post pages
4. **Slugification**: Post titles are converted to URL-friendly slugs for filenames

## Development Commands

**Install dependencies:**
```bash
uv sync
```

**Build the site:**
```bash
cd src && python build.py
```

**Repository Etiquette**:
- Write minimal lines of code for any change.
- Expect detailed instructions to make any changes. If user is not detailed in instructions, ask questions to gather more context.
- Use good variable conventions, don't use abbreviations, be expressive.
- Make a plan before coding, confirm with me if it sounds good.
- Add docstrings for every additional function you write.

## Dependencies

- Python 3.10+ (project uses Python 3.11.0)
- `mistune` for markdown processing (defined in pyproject.toml)
- `uv` package manager for dependency management

## File Structure

- `src/build.py` - Main build script
- `src/index.html` - Homepage template with embedded CSS
- `src/posts.json` - Blog post metadata
- `src/posts/` - Markdown content files
- `docs/` - Generated static site (build output)