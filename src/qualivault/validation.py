# -*- coding: utf-8 -*-
"""
QualiVault Transcript Validation Module

Uses Ollama (local LLM) to detect potential transcription errors and hallucinations
in interview transcripts. Validates CSV transcripts and flags suspicious patterns.
"""

import logging
import pandas as pd
from pathlib import Path
import requests
import json
from typing import Dict, List, Optional, Any
from collections import Counter

logger = logging.getLogger("QualiVault")


class OllamaValidator:
    """
    Validates transcripts using Ollama local LLM to detect:
    - Hallucinations (repeated phrases, nonsense)
    - Transcription errors (misheard words)
    - Audio artifacts (timestamps, metadata bleeding into text)
    - Language inconsistencies
    """
    
    def __init__(self, model="llama2", ollama_url="http://localhost:11434", timeout=60):
        """
        Initialize the validator.
        
        Args:
            model: Ollama model name (e.g., "llama2", "mistral", "llama3")
            ollama_url: URL where Ollama is running
            timeout: Request timeout in seconds (default 60, increase for large models)
        """
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        # Test connection
        if not self._check_ollama_connection():
            logger.warning("⚠️ Could not connect to Ollama. Make sure it's running.")
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _query_ollama(self, prompt: str, context: str = "", options: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a query to Ollama and get response.
        
        Args:
            prompt: The question/instruction to the LLM
            context: Additional context for the LLM
            options: Extra generation options (temperature, top_p, etc.)
            
        Returns:
            LLM response text
        """
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            base_options = {
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.9
            }
            if options:
                base_options.update(options)
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": base_options
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return ""
    
    def validate_segment(self, text: str, speaker: str, language: str = "Danish",
                        validation_params: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Validate a single transcript segment for errors.
        
        Args:
            text: The transcript text to validate
            speaker: Speaker identifier (for context)
            language: Expected language of the transcript
            
        Returns:
            Dictionary with validation results:
            {
                'has_issues': bool,
                'confidence': float (0-1),
                'issues': List[str],
                'suggestions': str
            }
        """
        default_prompt_template = """You are validating a {language} interview transcript segment.

Segment: "{text}"
Speaker: {speaker}

Analyze this segment and identify:
1. Hallucinations (repeated phrases, gibberish, random text)
2. Transcription errors (misheard words that don't fit context)
3. Audio artifacts (timestamps, file names, subtitles metadata)
4. Incomplete sentences or fragments

Respond in JSON format:
{{
    "has_issues": true/false,
    "confidence": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "suggestions": "brief correction suggestion"
}}

Be conservative - only flag clear problems."""

        prompt_template = default_prompt_template
        if validation_params:
            prompt_template = validation_params.get("prompt_template", default_prompt_template)

        prompt = prompt_template.format(language=language, text=text, speaker=speaker)

        options = None
        min_len = 10
        if validation_params:
            options = validation_params.get("ollama_options")
            min_len = validation_params.get("min_text_length", min_len)

        response = self._query_ollama(prompt, options=options)
        
        try:
            # Try to extract JSON from response
            if "{" in response:
                json_start = response.index("{")
                json_end = response.rindex("}") + 1
                result = json.loads(response[json_start:json_end])
            else:
                # Fallback if no JSON
                result = {
                    'has_issues': False,
                    'confidence': 0.5,
                    'issues': [],
                    'suggestions': ''
                }
        except Exception as e:
            logger.warning(f"Could not parse LLM response: {e}")
            result = {
                'has_issues': False,
                'confidence': 0.0,
                'issues': [],
                'suggestions': ''
            }
        
        return result
    
    def validate_transcript(self, csv_path: Path, sample_rate: float = 0.1,
                          language: str = "Danish",
                          validation_params: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Validate an entire transcript CSV.
        
        Args:
            csv_path: Path to the CSV transcript
            sample_rate: Fraction of segments to validate (0.0-1.0)
            language: Expected language
            
        Returns:
            Validation report dictionary
        """
        if not csv_path.exists():
            logger.error(f"CSV not found: {csv_path}")
            return {}
        
        # Allow sample_rate override from validation_params
        if validation_params:
            sample_rate = validation_params.get("sample_rate", sample_rate)
            min_len = validation_params.get("min_text_length", 10)
            max_segments = validation_params.get("max_segments")
        else:
            min_len = 10
            max_segments = None

        df = pd.read_csv(csv_path)
        logger.info(f"📋 Validating: {csv_path.name} ({len(df)} segments)")
        
        # Sample segments (validate every Nth segment)
        step = max(1, int(1 / sample_rate))
        sampled_indices = list(range(0, len(df), step))
        if max_segments:
            sampled_indices = sampled_indices[:max_segments]
        
        flagged_segments = []
        total_checked = 0
        
        for idx in sampled_indices:
            row = df.iloc[idx]
            text = str(row.get('Text', ''))
            speaker = str(row.get('Speaker', 'Unknown'))
            
            if len(text.strip()) < min_len:  # Skip very short segments
                continue
            
            total_checked += 1
            logger.info(f"  Checking segment {idx + 1}/{len(df)}...")
            
            validation = self.validate_segment(text, speaker, language, validation_params)
            
            if validation.get('has_issues'):
                flagged_segments.append({
                    'segment_index': idx,
                    'start': row.get('Start'),
                    'end': row.get('End'),
                    'speaker': speaker,
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'issues': validation.get('issues', []),
                    'confidence': validation.get('confidence', 0.0),
                    'suggestions': validation.get('suggestions', '')
                })
        
        report = {
            'csv_file': csv_path.name,
            'total_segments': len(df),
            'segments_checked': total_checked,
            'flagged_count': len(flagged_segments),
            'flagged_segments': flagged_segments,
            'validation_rate': sample_rate
        }
        
        logger.info(f"  ✅ Checked {total_checked} segments, flagged {len(flagged_segments)} issues")
        
        return report


def validate_recipe_transcripts(recipe_path: Path, transcripts_dir: Path,
                                sample_rate: float = 0.1,
                                model: str = "llama2",
                                language: str = "Danish",
                                ollama_url: str = "http://localhost:11434",
                                validation_params: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """
    Validate all transcribed interviews in a recipe.
    
    Args:
        recipe_path: Path to processing_recipe.yaml
        transcripts_dir: Directory containing CSV transcripts
        sample_rate: Fraction of segments to check per file
        model: Ollama model to use
        language: Expected language
        ollama_url: URL where Ollama is running
        
    Returns:
        List of validation reports
    """
    import yaml
    
    if not recipe_path.exists():
        logger.error(f"Recipe not found: {recipe_path}")
        return []
    
    with open(recipe_path, 'r') as f:
        recipe = yaml.safe_load(f)
    
    validator = OllamaValidator(model=model, ollama_url=ollama_url)
    reports = []
    
    for item in recipe:
        # Only validate transcribed items
        if item.get('transcribe_status') != 'transcribed':
            continue
        
        transcript_path = transcripts_dir / item['output_name'].replace('.flac', '.csv')
        
        if not transcript_path.exists():
            logger.warning(f"⚠️ Transcript not found: {transcript_path.name}")
            continue
        
        report = validator.validate_transcript(
            transcript_path,
            sample_rate=sample_rate,
            language=language,
            validation_params=validation_params
        )
        
        if report:
            report['interview_id'] = item['id']
            reports.append(report)
            
            # Update recipe with validation status
            item['validation_status'] = 'validated'
            item['flagged_issues'] = len(report.get('flagged_segments', []))
    
    # Save updated recipe
    with open(recipe_path, 'w') as f:
        yaml.dump(recipe, f, sort_keys=False, allow_unicode=True)
    
    logger.info(f"✅ Validated {len(reports)} transcripts")
    
    return reports


def extract_common_errors(validation_results: List[Dict], top_n: int = 20) -> List[Dict]:
    """
    Extract and count error TYPES from flagged segments.
    Groups errors by type (e.g., "Transcription error", "Hallucination", "Incomplete sentence")
    to identify systemic patterns for automation/correction.
    
    Args:
        validation_results: List of validation reports from validate_recipe_transcripts or validate_transcript
        top_n: Number of top error types to return
        
    Returns:
        List of dicts with 'error_type' and 'count' keys, sorted by frequency
    """
    error_counter = Counter()
    
    # Collect all errors from flagged segments
    for result in validation_results:
        for flagged_seg in result.get('flagged_segments', []):
            # Extract issues (which are error descriptions)
            issues = flagged_seg.get('issues', [])
            for issue in issues:
                issue_str = str(issue)
                
                # Extract just the error type (prefix before first ':')
                # E.g., "Transcription error: 'foo' might be..." → "Transcription error"
                if ':' in issue_str:
                    error_type = issue_str.split(':', 1)[0].strip()
                else:
                    error_type = issue_str
                
                error_counter[error_type] += 1
    
    # Convert to sorted list
    common_errors = [
        {'error_type': error_type, 'count': count}
        for error_type, count in error_counter.most_common(top_n)
    ]
    
    return common_errors
