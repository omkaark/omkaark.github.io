#!/usr/bin/env python3
"""
This file will take index.html, posts.json, styles.css, and scripts.js and build a static site in the dist directory.

I want a snappy website that is easy to build and deploy. The site should be simple, with a focus on content and performance. It should not use any frameworks or libraries that require a build step, so it can be easily hosted on any static file server.

The idea is if I have can write my /posts in markdown, I can convert them to HTML and serve them as static files. The build script should handle this conversion and copy the necessary files to the dist directory.
"""

import json
import re
import shutil
from pathlib import Path


def slugify(text):
    """Convert post title to filename format (lowercase, no punctuation, spaces to dashes)"""
    text = text.lower()
    # Remove punctuation except spaces and dashes
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace multiple spaces/dashes with single dash
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def markdown_to_html(markdown_text):
    """Convert markdown to HTML with basic formatting support"""
    lines = markdown_text.strip().split('\n')
    html_lines = []
    in_paragraph = False
    
    for line in lines:
        line = line.strip()
        
        # Handle headers
        if line.startswith('### '):
            if in_paragraph:
                html_lines.append('</p>')
                in_paragraph = False
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('## '):
            if in_paragraph:
                html_lines.append('</p>')
                in_paragraph = False
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('# '):
            if in_paragraph:
                html_lines.append('</p>')
                in_paragraph = False
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line == '':
            # Empty line - close paragraph if open
            if in_paragraph:
                html_lines.append('</p>')
                in_paragraph = False
        else:
            # Regular content line
            if not in_paragraph:
                html_lines.append('<p>')
                in_paragraph = True
            else:
                html_lines.append(' ')
            
            # Handle inline formatting
            processed_line = line
            
            # Bold text **text**
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_line)
            
            # Italic text *text*
            processed_line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_line)
            
            # Links [text](url)
            processed_line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', processed_line)
            
            html_lines.append(processed_line)
    
    # Close final paragraph if needed
    if in_paragraph:
        html_lines.append('</p>')
    
    return ''.join(html_lines)


def extract_css_from_index():
    """Extract CSS from index.html between STYLE comments"""
    with open('index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract CSS between /* STYLE */ comments
    css_match = re.search(r'/\* STYLE \*/(.*?)/\* STYLE \*/', content, re.DOTALL)
    if css_match:
        base_css = css_match.group(1).strip()
    else:
        # Fallback: extract from <style> tags
        style_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
        base_css = style_match.group(1).strip() if style_match else ""
    
    # Add post-specific CSS
    post_css = """
.post-content {
    margin-bottom: 80px;
    max-width: 800px;
}

.post-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 40px;
    padding-bottom: 20px;
    border-bottom: 1px solid #000;
}

.back-link {
    color: #000;
    text-decoration: none;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: bold;
}

.back-link:hover {
    text-decoration: underline;
}

.post-content h1 {
    margin: 30px 0 20px 0;
    font-size: 32px;
    line-height: 1.2;
}

.post-content h2 {
    margin: 30px 0 15px 0;
    font-size: 24px;
    line-height: 1.3;
}

.post-content h3 {
    margin: 25px 0 10px 0;
    font-size: 20px;
    line-height: 1.3;
}

.post-content p {
    margin-bottom: 20px;
    line-height: 1.6;
}

.post-content a {
    color: #000;
    text-decoration: underline;
}

.post-content a:hover {
    background: #f5f5dc;
}

@media (max-width: 768px) {
    .post-meta {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }
    
    .post-content h1 {
        font-size: 28px;
    }
}
"""
    
    return base_css + post_css


def build_index_page(posts):
    """Build the main index page with post list"""
    with open('index.html', 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Generate posts HTML
    posts_html = []
    for post in posts:
        slug = slugify(post['name'])
        posts_html.append(f'''
        <div class="post">
            <div class="post-title">
                <a href="{slug}.html" class="post-link">{post['name']}</a>
            </div>
            <div class="post-date">{post['date']}</div>
        </div>
        ''')
    
    # Replace posts placeholder
    posts_section = '\n'.join(posts_html)
    index_html = template.replace('<!-- POSTS -->', posts_section)
    
    return index_html


def build_post_page(post, css):
    """Build individual post page"""
    slug = slugify(post['name'])
    markdown_file = Path('posts') / f'{slug}.md'
    
    if not markdown_file.exists():
        print(f"Warning: {markdown_file} not found, skipping...")
        return None
    
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    html_content = markdown_to_html(markdown_content)
    
    post_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{post['name']} - Omkaar Kamath</title>
    <meta name="description" content="{post['name']} - Personal blog post">
    <style>
{css}
    </style>
</head>
<body>
    <header class="header">
        <h1 class="name">
            <a href="index.html" style="color: #000; text-decoration: none;">Omkaar Kamath</a>
        </h1>
        <div>
            <a href="https://twitter.com/omkizzy" class="social-link">Twitter</a>
            <a href="https://linkedin.com/in/omkaark" class="social-link">LinkedIn</a>
            <a href="https://github.com/omkaark" class="social-link">GitHub</a>
        </div>
    </header>
    
    <main class="post-content">
        <div class="post-meta">
            <a href="index.html" class="back-link">← Back</a>
            <div class="post-date">{post['date']}</div>
        </div>
        
        <article>
{html_content}
        </article>
    </main>
</body>
</html>'''
    
    return post_html


def build_site():
    """Main build function"""
    print("Building static site...")
    
    # Create/clean dist directory
    dist_dir = Path(Path(__file__).parent.parent, 'dist')
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    posts_dir = dist_dir / 'posts'
    if posts_dir.exists():
        shutil.rmtree(posts_dir)
    posts_dir.mkdir()
    
    # Read posts configuration
    try:
        with open('posts.json', 'r', encoding='utf-8') as f:
            posts = json.load(f)
    except FileNotFoundError:
        print("Error: posts.json not found")
        return False
    except json.JSONDecodeError:
        print("Error: Invalid JSON in posts.json")
        return False
    
    # Extract CSS
    css = extract_css_from_index()
    
    # Build index page
    try:
        index_html = build_index_page(posts)
        with open(dist_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(index_html)
        print(f"✓ Built index.html")
    except Exception as e:
        print(f"Error building index page: {e}")
        return False
    
    # Build individual post pages
    posts_built = 0
    for post in posts:
        try:
            post_html = build_post_page(post, css)
            if post_html:
                slug = slugify(post['name'])
                with open(posts_dir / f'{slug}.html', 'w', encoding='utf-8') as f:
                    f.write(post_html)
                print(f"✓ Built {slug}.html")
                posts_built += 1
        except Exception as e:
            print(f"Error building post '{post['name']}': {e}")
    
    print(f"\n🎉 Site built successfully!")
    print(f"   📄 Index page: dist/index.html")
    print(f"   📝 Posts built: {posts_built}/{len(posts)}")
    print(f"   📁 Output directory: dist/")
    
    return True


if __name__ == '__main__':
    success = build_site()
    if not success:
        exit(1)