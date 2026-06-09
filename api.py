# -*- coding: utf-8 -*-
"""
FastAPI REST API for Data Mapping Solution
Provides endpoints for comparing texts and processing policies
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Any
import logging
import os
import pandas as pd
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModel

from config import BERT_CONFIG, API_CONFIG, get_accuracy_level
from utils import (
    setup_logger, preprocess_text, lexical_overlap_score,
    validate_text_input, calculate_mapping_statistics
)

# Setup logging
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Data Mapping API",
    description="API for comparing texts and mapping policies",
    version="1.1.0"
)

# Global models (loaded once at startup)
tokenizer = None
model = None


# Pydantic models for request/response
class TextComparisonRequest(BaseModel):
    """Request model for text comparison"""
    text1: str
    text2: str


class TextComparisonResponse(BaseModel):
    """Response model for text comparison"""
    match: bool
    confidence: float
    similarity: dict
    processing_time_ms: float


class PolicyMappingRequest(BaseModel):
    """Request model for policy mapping"""
    policy_name: str
    policy_text: str


class PolicyMappingResponse(BaseModel):
    """Response model for policy mapping"""
    policy_name: str
    mapped_section: str
    confidence: int
    accuracy: str
    rationale: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    models_loaded: bool


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global tokenizer, model
    
    try:
        logger.info("Loading BERT model on startup...")
        config = BERT_CONFIG
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        model = AutoModel.from_pretrained(config['model_name'])
        model.to(config['device'])
        model.eval()
        logger.info("BERT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global tokenizer, model
    if model:
        del model
        logger.info("Model unloaded")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint
    
    Returns:
        HealthResponse with status and model information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=(tokenizer is not None and model is not None)
    )


# Text comparison endpoint
@app.post("/compare", response_model=TextComparisonResponse)
async def compare_texts(request: TextComparisonRequest):
    """Compare two texts and return similarity metrics
    
    Args:
        request: TextComparisonRequest with text1 and text2
        
    Returns:
        TextComparisonResponse with match status and confidence
        
    Raises:
        HTTPException: If texts are invalid or processing fails
    """
    # Validate inputs
    is_valid1, msg1 = validate_text_input(request.text1)
    is_valid2, msg2 = validate_text_input(request.text2)
    
    if not is_valid1 or not is_valid2:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {msg1} {msg2}"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Preprocess texts
        processed1 = preprocess_text(request.text1)
        processed2 = preprocess_text(request.text2)
        
        # Calculate lexical similarity
        lexical_sim = lexical_overlap_score(processed1, processed2)
        
        # Calculate semantic similarity with BERT
        if tokenizer and model:
            with torch.no_grad():
                inputs1 = tokenizer(request.text1, return_tensors="pt", truncation=True, padding=True)
                inputs2 = tokenizer(request.text2, return_tensors="pt", truncation=True, padding=True)
                
                outputs1 = model(**inputs1)
                outputs2 = model(**inputs2)
                
                emb1 = outputs1.last_hidden_state[:, 0, :]
                emb2 = outputs2.last_hidden_state[:, 0, :]
                
                semantic_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        else:
            semantic_sim = 0.0
        
        # Combine similarities
        combined_score = (semantic_sim * 0.8) + (lexical_sim * 0.2)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return TextComparisonResponse(
            match=combined_score > 0.5,
            confidence=min(combined_score, 1.0),
            similarity={
                "semantic": float(semantic_sim),
                "lexical": float(lexical_sim),
                "combined": float(combined_score)
            },
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error comparing texts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


# Batch comparison endpoint
@app.post("/compare-batch")
async def compare_batch(requests: List[TextComparisonRequest]):
    """Compare multiple text pairs
    
    Args:
        requests: List of TextComparisonRequest objects
        
    Returns:
        List of TextComparisonResponse objects
        
    Raises:
        HTTPException: If processing fails
    """
    if not requests:
        raise HTTPException(status_code=400, detail="Empty request list")
    
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 comparisons per request")
    
    results = []
    for req in requests:
        try:
            result = await compare_texts(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            results.append({"error": str(e)})
    
    return {"results": results, "count": len(results)}


# Policy mapping endpoint
@app.post("/map-policy", response_model=PolicyMappingResponse)
async def map_policy(request: PolicyMappingRequest):
    """Map a policy to a reference section
    
    Args:
        request: PolicyMappingRequest with policy name and text
        
    Returns:
        PolicyMappingResponse with mapping details
        
    Raises:
        HTTPException: If processing fails
    """
    # Validate input
    is_valid, msg = validate_text_input(request.policy_text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid policy text: {msg}")
    
    try:
        # This is a simplified version - full implementation would use the mapper
        processed_text = preprocess_text(request.policy_text)
        
        # For now, return a placeholder response
        confidence = int(len(processed_text.split()) * 10) % 100
        
        return PolicyMappingResponse(
            policy_name=request.policy_name,
            mapped_section="Reference Section",
            confidence=confidence,
            accuracy=get_accuracy_level(confidence),
            rationale="Policy mapping endpoint - requires full implementation"
        )
    
    except Exception as e:
        logger.error(f"Error mapping policy: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


# Statistics endpoint
@app.get("/statistics")
async def get_statistics(filepath: Optional[str] = None):
    """Get statistics from a CSV mapping results file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Dictionary with mapping statistics
        
    Raises:
        HTTPException: If file not found or invalid
    """
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        df = pd.read_csv(filepath)
        stats = calculate_mapping_statistics(df)
        return {
            "statistics": stats,
            "file": filepath,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reading statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Data Mapping API",
        "version": "1.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "compare": "/compare",
            "compare_batch": "/compare-batch",
            "map_policy": "/map-policy",
            "statistics": "/statistics",
            "docs": "/docs"
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Any, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    config = API_CONFIG
    logger.info(f"Starting API on {config['host']}:{config['port']}")
    
    uvicorn.run(
        app,
        host=config['host'],
        port=config['port'],
        reload=config['reload'],
        log_level=config['log_level']
    )
