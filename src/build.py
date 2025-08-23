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
import mistune


def slugify(text):
    """Convert post title to filename format (lowercase, no punctuation, spaces to dashes)"""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text) # remove punctuation except spaces and dashes
    text = re.sub(r'[-\s]+', '-', text) # replace spaces/dashes with single dash
    return text.strip('-')


def markdown_to_html(markdown_text):
    """Convert markdown to HTML with basic formatting support"""
    lines = markdown_text.strip().split('\n')
    html_lines = []
    in_paragraph = False
    
    for line in lines:
        line = line.strip()
        
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
        elif line.startswith('# '):
            if in_paragraph:
                html_lines.append('</p>')
                in_paragraph = False
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line == '': # empty line - close paragraph if open
            if in_paragraph:
                html_lines.append('</p>')
                in_paragraph = False
        else:
            if not in_paragraph:
                html_lines.append('<p>')
                in_paragraph = True
            else:
                html_lines.append(' ')
            
            # handle inline formatting
            processed_line = line
            
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_line) # bold text **text**
            
            processed_line = re.sub(r'\*(.*?)\*', r'<em>\1</em>', processed_line) # italic text *text*
            
            processed_line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', processed_line) # links [text](url)
            
            html_lines.append(processed_line)
    
    # close final paragraph if needed
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
    font-size: 20px;
    font-family: Monaco, monospace;
}

.post-content a {
    color: #000;
    text-decoration: underline;
}

.post-content img {
    margin: 20px 0 20px 0;
    max-width: 100%;
    max-height: 400px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    display: block;
    margin-left: auto;
    margin-right: auto;
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

    .post-content {
        margin: 0px 10px;
    }
    
    .post-content h1 {
        font-size: 28px;
    }

    .post-content p {
        font-family: Monaco, monospace;
    }
}

/* Inline code */
.post-content :not(pre) > code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: .95em;
  background: rgba(127,127,127,.12);
  border: 1px solid rgba(127,127,127,.25);
  padding: .15em .4em;
  border-radius: .35em;
  word-break: break-word;
}

/* Code blocks */
.post-content pre {
  margin: 20px 0;
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid rgba(127,127,127,.25);
  background: #f7f7f9;
  overflow: auto;
  -webkit-overflow-scrolling: touch;
  tab-size: 2;
}

.post-content pre code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 14px;
  line-height: 1.55;
  display: block;
  white-space: pre;          /* keep formatting */
}

"""
    
    return base_css + post_css


def build_index_page(posts):
    """Build the main index page with post list"""
    with open('index.html', 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Generate posts HTML
    posts_html = []
    for post in reversed(posts):
        slug = slugify(post['name'])
        posts_html.append(f'''
        <a href="posts/{slug}.html" class="post-link">
            <div class="post">
                <div class="post-title">
                    {post['name']}
                </div>
                <div class="post-date">{post['date']}</div>
            </div>
        </a>
        ''')
    
    # Replace posts placeholder
    posts_section = '\n'.join(posts_html)
    index_html = template.replace('<!-- POSTS -->', posts_section)
    
    return index_html


def build_post_page(idx, post, css):
    """Build individual post page"""
    slug = slugify(post['name'])
    markdown_file = Path('posts') / f'{idx}.md'
    
    if not markdown_file.exists():
        print(f"Warning: {markdown_file} not found, skipping...")
        return None
    
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    html_content = mistune.html(markdown_content)
    
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
        <a href="https://omkaark.github.io" style="text-decoration: none;" class="social-link"><h1 class="name">Omkaar Kamath</h1></a>
        <div>
            <a href="https://twitter.com/omkizzy" class="social-link">Twitter</a>
            <a href="https://linkedin.com/in/omkaark" class="social-link">LinkedIn</a>
            <a href="https://github.com/omkaark" class="social-link">GitHub</a>
            <a href="https://youtube.com/@omkizzy" class="social-link">Youtube</a>
        </div>
    </header>
    
    <main class="post-content">
{html_content}
    </main>
</body>
</html>'''
    
    return post_html


def generate_sitemap(posts, output_directory):
    """Generate XML sitemap for the website"""
    base_url = "https://omkaark.com"
    
    sitemap_content = ['<?xml version="1.0" encoding="UTF-8"?>']
    sitemap_content.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    
    # Add homepage
    sitemap_content.append('  <url>')
    sitemap_content.append(f'    <loc>{base_url}/</loc>')
    sitemap_content.append('    <changefreq>weekly</changefreq>')
    sitemap_content.append('    <priority>1.0</priority>')
    sitemap_content.append('  </url>')
    
    # Add post pages
    for post in posts:
        slug = slugify(post['name'])
        post_date = post['date']  # Format: YYYY-MM-DD
        
        sitemap_content.append('  <url>')
        sitemap_content.append(f'    <loc>{base_url}/posts/{slug}.html</loc>')
        sitemap_content.append(f'    <lastmod>{post_date}</lastmod>')
        sitemap_content.append('    <changefreq>monthly</changefreq>')
        sitemap_content.append('    <priority>0.8</priority>')
        sitemap_content.append('  </url>')
    
    sitemap_content.append('</urlset>')
    
    # Write sitemap to file
    sitemap_xml = '\n'.join(sitemap_content)
    sitemap_path = output_directory / 'sitemap.xml'
    
    with open(sitemap_path, 'w', encoding='utf-8') as f:
        f.write(sitemap_xml)
    
    return sitemap_path


def build_site():
    """Main build function"""
    print("Building static site...")
    
    # Create/clean dist directory
    dist_dir = Path(Path(__file__).parent.parent, 'docs')
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    posts_dir = dist_dir / 'posts'
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
        print(f"-> Built index.html")
    except Exception as e:
        print(f"Error building index page: {e}")
        return False
    
    # Build individual post pages
    posts_built = 0
    for idx, post in enumerate(posts, start=1):
        try:
            post_html = build_post_page(idx, post, css)
            if post_html:
                slug = slugify(post['name'])
                with open(posts_dir / f'{slug}.html', 'w', encoding='utf-8') as f:
                    f.write(post_html)
                print(f"-> Built posts/{slug}.html")
                posts_built += 1
        except Exception as e:
            print(f"Error building post '{post['name']}': {e}")
    
    # Generate sitemap
    try:
        generate_sitemap(posts, dist_dir)
        print(f"-> Built sitemap.xml")
    except Exception as e:
        print(f"Error generating sitemap: {e}")
        return False
    
    print()
    print(f"- Site built successfully!")
    print(f"- Index page: docs/index.html")
    print(f"- Posts built: {posts_built}/{len(posts)}")
    print(f"- Sitemap: docs/sitemap.xml")
    print(f"- Output directory: docs/")
    
    return True


if __name__ == '__main__':
    success = build_site()
    if not success:
        exit(1)