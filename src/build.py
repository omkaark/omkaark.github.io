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

def get_image_link(image_name):
    """Image name to github raw link"""
    return f"https://raw.githubusercontent.com/omkaark/omkaark.github.io/refs/heads/main/public/{image_name}?raw=true"

def get_css():
    """Get CSS from style.css"""
    with open('style.css', 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content

def get_head_tags(name: str = 'index', image_name = 'omkaark.jpeg'):
    """Get head tags for SEO"""
    if name == 'index':
        head_tags = f'''
            <link rel="icon" type="image/x-icon" href="{get_image_link('compass.jpg')}">
            <title>Omkaar's Recipes</title>
            <meta name="description" content="These are all my thoughts that I want google and chatgpt to index.">
            <meta property="og:title" content="Omkaar's Recipes">
            <meta property="og:description" content="These are all my thoughts that I want google and chatgpt to index.">
            <meta property="og:type" content="website">
            <meta property="og:url" content="https://omkaark.com">
            <meta property="og:image" content="{get_image_link(image_name)}">
            <meta name="twitter:card" content="summary_large_image">
            <meta name="twitter:title" content="Omkaar's Recipes">
            <meta name="twitter:description" content="These are all my thoughts that I want google and chatgpt to index.">
            <meta name="twitter:image" content="{get_image_link(image_name)}">
        '''
    else:
        head_tags = f'''
            <link rel="icon" type="image/x-icon" href="{get_image_link('compass.jpg')}">
            <title>{name} - Omkaar Kamath</title>
            <meta name="description" content="{name} - Personal blog post">
            <meta property="og:title" content="{name} - Omkaar Kamath">
            <meta property="og:description" content="{name} - Personal blog post">
            <meta property="og:type" content="article">
            <meta property="og:url" content="https://omkaark.com/posts/{slugify(name)}.html">
            <meta property="og:image" content="{get_image_link(image_name)}">
            <meta name="twitter:card" content="summary_large_image">
            <meta name="twitter:title" content="{name} - Omkaar Kamath">
            <meta name="twitter:description" content="{name} - Personal blog post">
            <meta name="twitter:image" content="{get_image_link(image_name)}">
        '''
    return head_tags

def build_index_page(posts):
    """Build the main index page with post list"""
    with open('template.html', 'r', encoding='utf-8') as f:
        template = f.read()

    # Replace styles
    css = get_css()
    index_html = template.replace('/* STYLE */', css)

    # Replace SEO meta tags
    head_tags = get_head_tags()
    index_html = index_html.replace('<!-- SEO -->', head_tags)

    # Replace content with posts
    posts_html = ['<main class="posts" id="posts-container">']
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
    posts_html.append('</main>')
    posts_section = '\n'.join(posts_html)
    index_html = index_html.replace('<!-- CONTENT -->', posts_section)
    
    return index_html


def build_post_page(idx, post, css):
    """Build individual post page"""
    slug = slugify(post['name'])

    with open('template.html', 'r', encoding='utf-8') as f:
        template = f.read()

    # Replace styles
    css = get_css()
    index_html = template.replace('/* STYLE */', css)

    # Replace SEO meta tags
    head_tags = get_head_tags(post['name'], image_name=post.get('image', 'omkaark.jpeg'))
    index_html = index_html.replace('<!-- SEO -->', head_tags)

    # Write post content
    markdown_file = Path('posts') / f'{idx}.md'
    
    if not markdown_file.exists():
        print(f"Warning: {markdown_file} not found, skipping...")
        return None
    
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    html_content = mistune.html(markdown_content)
    post_html = f'''
    <main class="post-content">
        {html_content}
    </main>
    '''
    index_html = index_html.replace('<!-- CONTENT -->', post_html)
    
    return index_html


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
    css = get_css()
    
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
    
    # Write CNAME
    try:
        cname_path = dist_dir / 'CNAME'
        with open(cname_path, 'w', encoding='utf-8') as f:
            f.write('omkaark.com\n')
        print(f"-> Wrote CNAME")
    except Exception as e:
        print(f"Error writing CNAME: {e}")
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