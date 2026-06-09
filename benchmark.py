# -*- coding: utf-8 -*-
"""
Performance Benchmarking Suite for Data Mapping Solution
Measures and reports on mapping efficiency and quality
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from config import HYBRID_MAPPER_CONFIG, PDF_POLICIES
from utils import (
    setup_logger, preprocess_text, lexical_overlap_score,
    calculate_mapping_statistics
)

logger = setup_logger(__name__)


class PerformanceBenchmark:
    """Benchmark class for measuring mapping performance"""
    
    def __init__(self):
        """Initialize benchmark tracker"""
        self.results: List[Dict] = []
        self.timings: Dict[str, float] = {}
    
    def measure_preprocessing(self, texts: List[str]) -> float:
        """Measure time to preprocess texts
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            Time taken in seconds
        """
        start = time.time()
        for text in texts:
            preprocess_text(text)
        elapsed = time.time() - start
        
        self.timings['preprocessing'] = elapsed
        logger.info(f"Preprocessing {len(texts)} texts: {elapsed:.4f}s")
        return elapsed
    
    def measure_similarity_scoring(self, text_pairs: List[Tuple[str, str]]) -> float:
        """Measure time to calculate similarity scores
        
        Args:
            text_pairs: List of text pairs
            
        Returns:
            Time taken in seconds
        """
        start = time.time()
        for text1, text2 in text_pairs:
            lexical_overlap_score(text1, text2)
        elapsed = time.time() - start
        
        self.timings['similarity'] = elapsed
        logger.info(f"Similarity scoring {len(text_pairs)} pairs: {elapsed:.4f}s")
        return elapsed
    
    def measure_pdf_preparation(self) -> float:
        """Measure time to prepare PDF policy descriptions
        
        Returns:
            Time taken in seconds
        """
        start = time.time()
        pdf_descriptions = list(PDF_POLICIES.values())
        for desc in pdf_descriptions:
            preprocess_text(desc)
        elapsed = time.time() - start
        
        self.timings['pdf_prep'] = elapsed
        logger.info(f"PDF preparation: {elapsed:.4f}s")
        return elapsed
    
    def measure_csv_operations(self, size: int = 1000) -> Dict[str, float]:
        """Measure CSV read/write operations
        
        Args:
            size: Number of rows to test
            
        Returns:
            Dictionary with timings for read and write
        """
        # Create test DataFrame
        test_df = pd.DataFrame({
            'Policy ID': [f'P-{i}' for i in range(size)],
            'Policy Name': [f'Policy {i}' for i in range(size)],
            'Description': [f'Description for policy {i}' for i in range(size)],
        })
        
        # Measure write
        start = time.time()
        test_df.to_csv('/tmp/test_output.csv', index=False)
        write_time = time.time() - start
        
        # Measure read
        start = time.time()
        pd.read_csv('/tmp/test_output.csv')
        read_time = time.time() - start
        
        timings = {'write': write_time, 'read': read_time}
        self.timings['csv_write'] = write_time
        self.timings['csv_read'] = read_time
        
        logger.info(f"CSV write ({size} rows): {write_time:.4f}s")
        logger.info(f"CSV read ({size} rows): {read_time:.4f}s")
        
        return timings
    
    def measure_batch_processing(self, batch_sizes: List[int]) -> Dict[int, float]:
        """Measure processing time for different batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary mapping batch size to processing time
        """
        results = {}
        
        for size in batch_sizes:
            # Create test data
            texts = [f'Test policy {i}' for i in range(size)]
            
            # Measure processing
            start = time.time()
            for text in texts:
                preprocess_text(text)
                for pdf_desc in list(PDF_POLICIES.values())[:5]:  # Sample 5 policies
                    lexical_overlap_score(text, pdf_desc)
            elapsed = time.time() - start
            
            results[size] = elapsed
            throughput = size / elapsed if elapsed > 0 else 0
            logger.info(f"Batch size {size}: {elapsed:.4f}s ({throughput:.1f} items/sec)")
        
        return results
    
    def measure_mapping_quality(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Measure quality metrics of mapping results
        
        Args:
            predictions: DataFrame with mapping results
            
        Returns:
            Dictionary with quality metrics
        """
        if 'Accuracy' not in predictions.columns:
            logger.warning("No Accuracy column in predictions")
            return {}
        
        stats = calculate_mapping_statistics(predictions)
        
        # Calculate additional metrics
        high_count = len(predictions[predictions['Accuracy'] == 'High'])
        total_count = len(predictions)
        high_accuracy_pct = (high_count / total_count * 100) if total_count > 0 else 0
        
        quality_metrics = {
            'average_confidence': stats.get('average_confidence', 0),
            'high_accuracy_percentage': high_accuracy_pct,
            'median_confidence': stats.get('median_confidence', 0),
            'std_dev': stats.get('std_dev', 0),
        }
        
        return quality_metrics
    
    def generate_report(self) -> str:
        """Generate performance benchmark report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*70)
        report.append("PERFORMANCE BENCHMARK REPORT")
        report.append("="*70)
        
        report.append("\nTiming Measurements:")
        report.append("-"*70)
        total_time = 0
        for operation, time_sec in self.timings.items():
            report.append(f"  {operation:.<40} {time_sec:>10.4f}s")
            total_time += time_sec
        
        report.append("-"*70)
        report.append(f"  {'Total':.<40} {total_time:>10.4f}s")
        
        report.append("\n" + "="*70)
        return "\n".join(report)
    
    def print_report(self):
        """Print the performance report"""
        print(self.generate_report())


def run_benchmarks():
    """Run all performance benchmarks"""
    
    logger.info("Starting performance benchmarks...")
    
    benchmark = PerformanceBenchmark()
    
    # 1. Preprocessing benchmark
    logger.info("\n[1/4] Preprocessing Benchmark")
    test_texts = [
        "Employee authentication requirement",
        "Data privacy and security measures",
        "Network access control policy",
        "Incident response procedures"
    ] * 100  # 400 texts
    benchmark.measure_preprocessing(test_texts)
    
    # 2. Similarity scoring benchmark
    logger.info("\n[2/4] Similarity Scoring Benchmark")
    test_pairs = [
        ("Employee must authenticate", "Multi-factor authentication required"),
        ("Data privacy policy", "Information security standards"),
    ] * 100  # 200 pairs
    benchmark.measure_similarity_scoring(test_pairs)
    
    # 3. PDF preparation benchmark
    logger.info("\n[3/4] PDF Preparation Benchmark")
    benchmark.measure_pdf_preparation()
    
    # 4. CSV operations benchmark
    logger.info("\n[4/4] CSV Operations Benchmark")
    benchmark.measure_csv_operations(size=1000)
    
    # 5. Batch processing benchmark
    logger.info("\n[5/5] Batch Processing Benchmark")
    batch_results = benchmark.measure_batch_processing([100, 500, 1000])
    
    # Print report
    benchmark.print_report()
    
    # Print batch processing results
    logger.info("\nBatch Processing Results:")
    logger.info("-"*70)
    for size, time_sec in batch_results.items():
        throughput = size / time_sec if time_sec > 0 else 0
        logger.info(f"  Size: {size:>5}, Time: {time_sec:>8.4f}s, Throughput: {throughput:>8.1f} items/sec")
    
    logger.info("\nBenchmarks completed!")


if __name__ == '__main__':
    run_benchmarks()
