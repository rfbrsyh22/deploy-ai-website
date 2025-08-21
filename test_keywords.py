#!/usr/bin/env python3
"""
Test script to check keyword counts
"""

import index

print("✅ Backend loaded successfully!")
print()
print("📊 Keyword Analysis:")
print(f"  - Legitimate indicators: {len(index.INDONESIAN_KEYWORDS['legitimate_indicators'])} words")
print(f"  - Suspicious indicators: {len(index.INDONESIAN_KEYWORDS['suspicious_indicators'])} words") 
print(f"  - Neutral keywords: {len(index.INDONESIAN_KEYWORDS['neutral_keywords'])} words")

total_keywords = (
    len(index.INDONESIAN_KEYWORDS['legitimate_indicators']) + 
    len(index.INDONESIAN_KEYWORDS['suspicious_indicators']) + 
    len(index.INDONESIAN_KEYWORDS['neutral_keywords'])
)

print(f"  - Total keywords: {total_keywords} words")
print()

# Test enhanced word fixes by counting patterns in the function
print("🔧 Enhanced Indonesian Word Fixes:")
print("  - Counting OCR fix patterns in source code...")

# Count patterns by reading the source file directly
with open('backend_working.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find word_fixes dictionary
word_fixes_start = content.find('word_fixes = {')
word_fixes_end = content.find('\n    }', word_fixes_start)
word_fixes_section = content[word_fixes_start:word_fixes_end]

# Count patterns (each line with ':' is a pattern)
pattern_count = word_fixes_section.count("': '") + word_fixes_section.count('": "')
print(f"  - Total OCR fix patterns: {pattern_count} patterns")
print()

print("🎯 Target Achievement:")
print(f"  - Target keywords: 2000+ words")
print(f"  - Current keywords: {total_keywords} words")
print(f"  - Achievement: {'✅ ACHIEVED' if total_keywords >= 2000 else '⚠️ NEED MORE'}")
print()

print(f"  - Target OCR fixes: 1000+ patterns")
print(f"  - Current OCR fixes: {pattern_count} patterns")
print(f"  - Achievement: {'✅ ACHIEVED' if pattern_count >= 1000 else '⚠️ NEED MORE'}")
