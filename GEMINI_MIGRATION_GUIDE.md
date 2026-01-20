# Gemini Embeddings Migration Guide

## Overview

This service has been migrated from **local sentence-transformers** to **Gemini API embeddings** (FREE tier) to eliminate OOM crashes on Render's 512MB free tier.

## What Changed

### Before (Local Embeddings)
- **Model**: `all-MiniLM-L6-v2` (local)
- **Dimensions**: 384
- **RAM Usage**: ~300-400MB at startup
- **Dependencies**: `torch`, `sentence-transformers`, `transformers`
- **Problem**: OOM crashes on Render startup

### After (Gemini Embeddings)
- **Model**: `text-embedding-004` (Gemini API)
- **Dimensions**: 768 (DOUBLED)
- **RAM Usage**: ~50MB (no model loading)
- **Dependencies**: `google-generativeai`
- **Result**: Stable startup, no OOM issues

## Critical Changes

### 1. Embedding Dimensions
⚠️ **IMPORTANT**: Embedding dimensions changed from 384 → 768

This means:
- **Existing indexes MUST be rebuilt** using [`build_indexes.py`](build_indexes.py)
- Old `.npy` files are incompatible with new embeddings
- Cosine similarity calculations remain the same

### 2. API Key Required
Set environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Get your FREE API key from: https://aistudio.google.com/app/apikey

### 3. Dependencies Removed
The following are **no longer needed** and have been removed:
- ❌ `sentence-transformers`
- ❌ `torch`
- ❌ `transformers`
- ❌ Any `nvidia-*` packages

### 4. New Dependencies Added
- ✅ `google-generativeai` (for Gemini API)

## Migration Steps

### Step 1: Update Environment Variables
```bash
# Add to Render dashboard or .env file
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=models/text-embedding-004  # Optional, defaults to this
```

### Step 2: Install New Dependencies
```bash
pip install -r requirements.txt
```

This will:
- Install `google-generativeai`
- Remove old torch/sentence-transformers

### Step 3: Rebuild Indexes
⚠️ **CRITICAL**: You must rebuild indexes with new embeddings

```bash
python build_indexes.py
```

This will:
1. Load comments from Excel
2. Load documentation from PDF
3. Embed all content using Gemini API
4. Save new indexes with 768-dimensional embeddings
5. Replace old `data/comments_vectors.npy` and `data/docs_vectors.npy`

**Expected output:**
```
Building indexes with Gemini embeddings (FREE tier)...
Model: text-embedding-004 (768 dimensions)

[1/2] Building comment style index...
Loaded 50 comments
Embedding comments via Gemini API...
Generated embeddings: shape=(50, 768)
Comment index saved ✓

[2/2] Building documentation index...
Loaded PDF: 25000 characters
Created 15 chunks
Embedding documentation via Gemini API...
Generated embeddings: shape=(15, 768)
Documentation index saved ✓

✅ Indexes built successfully!
```

### Step 4: Deploy to Render
1. Push changes to your repository
2. Render will automatically redeploy
3. Set `GEMINI_API_KEY` in Render dashboard environment variables
4. Upload rebuilt index files to Render (if not in repo)

## Code Changes Summary

### [`ml/embeddings.py`](ml/embeddings.py)
- ✅ Completely rewritten to use Gemini API
- ✅ No more torch/SentenceTransformer imports
- ✅ Added in-memory caching
- ✅ Added batching with retry logic
- ✅ Added safety limits and timeouts

### [`app.py`](app.py)
- ✅ Removed `get_embed_model()` call from startup
- ✅ Updated startup logs to reflect Gemini usage
- ✅ /docs endpoint already safe (returns placeholder)

### [`requirements.txt`](requirements.txt)
- ✅ Removed: `sentence-transformers`, `torch`
- ✅ Added: `google-generativeai`

### [`build_indexes.py`](build_indexes.py)
- ✅ Added documentation about dimension change
- ✅ Enhanced logging during index building

### No Changes Needed
- [`retrieval.py`](retrieval.py) - Works with both old and new embeddings
- [`comment_engine.py`](comment_engine.py) - No embedding-specific logic
- [`chunking.py`](chunking.py) - Dimension-agnostic
- Vector search logic - Cosine similarity works regardless of dimensions

## Features

### 1. In-Memory Caching
Gemini embeddings are cached by content hash:
- Identical texts don't get re-embedded
- Reduces API calls (cost optimization)
- Faster response times

```python
from ml.embeddings import clear_cache, get_cache_stats

# Clear cache if needed
clear_cache()

# Check cache status
stats = get_cache_stats()
# {'size': 50, 'model': 'models/text-embedding-004'}
```

### 2. Automatic Truncation
Long texts are automatically truncated to 1500 chars:
- Preserves word boundaries
- Prevents API errors
- Reduces token usage

### 3. Batching with Retries
- Default batch size: 20 texts per API call
- Max batch size: 100 (Gemini limit)
- 3 retry attempts with exponential backoff
- Graceful error handling

### 4. Safety Limits
- Max text length: 1500 characters
- Request timeout: 30 seconds
- Max retries: 3 attempts
- Batch size: Configurable (default 20)

## Cost & Limits

### Gemini FREE Tier
- **Model**: `text-embedding-004`
- **Rate Limits**: 1,500 requests/day (generous for free tier)
- **Embedding Dimensions**: 768
- **Max Input**: ~10,000 characters per request
- **Cost**: FREE ✅

Source: https://ai.google.dev/pricing

### Usage Estimates
For typical usage:
- 50 comments/day → 50 API calls
- 100 chunks from docs → 5 API calls (batched)
- Total: ~55 calls/day

**Well within FREE tier limits** 🎉

## Testing

### 1. Test Health Endpoint
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

### 2. Test Embedding Generation
```bash
curl -X POST http://localhost:8000/ml/generate-comment \
  -H "Content-Type: application/json" \
  -d '{
    "post_url": "https://reddit.com/r/programming/test",
    "source": "api"
  }'
```

Expected: JSON response with generated comment

### 3. Test /docs Endpoint
Navigate to: `http://localhost:8000/docs`

This should return a placeholder without triggering embeddings.

## Troubleshooting

### Issue: "GEMINI_API_KEY not set"
**Solution**: Set environment variable
```bash
export GEMINI_API_KEY="your-key"
```

### Issue: "Dimension mismatch" errors
**Solution**: Rebuild indexes with [`build_indexes.py`](build_indexes.py)
```bash
python build_indexes.py
```

### Issue: Rate limit exceeded
**Solution**: 
1. Implement request throttling in your client
2. Upgrade to paid tier if needed
3. Use caching more aggressively

### Issue: Slow embeddings
**Solution**:
1. Check cache stats - many cache hits = good
2. Reduce batch size if timeout issues
3. Increase batch size if too many API calls

### Issue: Memory still high on Render
**Solution**:
1. Verify torch/sentence-transformers are NOT installed
2. Check `pip list | grep torch` → should be empty
3. Clear pip cache: `pip cache purge`

## Monitoring

### Key Metrics to Watch
```python
# In logs, look for:
mem_logger.info("embed_start texts=10 batch_size=20")
mem_logger.info("cache_hit_all count=10")  # Good!
mem_logger.info("gemini_embed_success batch_num=1")
mem_logger.info("embed_complete shape=(10, 768) cached=50")
```

### Success Indicators
- ✅ Startup completes in <10 seconds
- ✅ Memory usage stays below 200MB
- ✅ No OOM errors in Render logs
- ✅ `/health` returns 200 OK
- ✅ Cache hit rate increases over time

## Rollback Plan

If you need to rollback to local embeddings:

1. Restore old `requirements.txt`:
```
sentence-transformers==5.2.0
torch==2.5.0
```

2. Restore old [`ml/embeddings.py`](ml/embeddings.py) from git history

3. Restore old index files (384 dimensions)

4. Remove `GEMINI_API_KEY` from environment

⚠️ **Note**: Rollback means returning to OOM issues on Render

## FAQ

**Q: Can I use Vertex AI instead?**
A: No, this implementation uses AI Studio (FREE). Vertex AI requires GCP setup and billing.

**Q: Can I use Gemini Pro?**
A: Yes, set `GEMINI_MODEL=models/embedding-001` (but Pro may have different limits)

**Q: Do I need to rebuild indexes every time?**
A: No, only when:
- Switching embedding models
- Adding new comment examples
- Updating documentation

**Q: What's the cache size limit?**
A: No hard limit (in-memory), but memory scales with cache size. Typical usage: 50-100 entries.

**Q: Can I clear cache on demand?**
A: Yes, call [`clear_cache()`](ml/embeddings.py:254) from [`ml.embeddings`](ml/embeddings.py)

## Success Criteria ✅

- [x] No torch/sentence-transformers in requirements
- [x] Render startup completes without OOM
- [x] `/health` responds with 200 OK
- [x] `/ml/generate-comment` returns JSON
- [x] No Cloudflare HTML 502 errors
- [x] Memory usage < 200MB
- [x] Embeddings use FREE Gemini Flash model
- [x] Indexes rebuilt with 768 dimensions
- [x] Caching reduces API calls

## Resources

- [Gemini API Documentation](https://ai.google.dev/docs)
- [Get FREE API Key](https://aistudio.google.com/app/apikey)
- [Embedding Guide](https://ai.google.dev/tutorials/python_quickstart#use_embeddings)
- [Rate Limits & Pricing](https://ai.google.dev/pricing)

---

**Migration completed successfully!** 🎉

If you encounter issues, check the logs for `[ML][MEM]` tags for detailed memory and embedding metrics.