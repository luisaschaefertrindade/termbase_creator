# Term Base Creator
# Copyright (c) 2025 Luísa Schaefer Trindade
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import spacy
from collections import Counter, defaultdict
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from docx import Document
from spacy.matcher import Matcher
from spacy.symbols import ORTH
from spacy.lang.en.stop_words import STOP_WORDS
import threading
import itertools
from math import log

# -------------------- Utilities --------------------

def chunk_text(text, max_chars=500_000):
    """Split text into manageable chunks for processing."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at paragraph boundaries
        boundary = text.rfind("\n\n", start, end)
        if boundary <= start:
            # Try to break at sentence boundaries
            boundary = text.rfind(". ", start, end)
            if boundary <= start:
                boundary = end
        chunks.append(text[start:boundary].strip())
        start = boundary
    return [chunk for chunk in chunks if chunk.strip()]

def clean_term(term):
    """Clean terms by removing leading/trailing stopwords and extra whitespace."""
    tokens = term.split()
    # Remove leading stopwords (but preserve honorifics + name combinations)
    while tokens and tokens[0].lower() in STOP_WORDS and not tokens[0].lower() in HONORIFICS:
        tokens.pop(0)
    # Remove trailing stopwords
    while tokens and tokens[-1].lower() in STOP_WORDS:
        tokens.pop()
    return " ".join(tokens).strip()

HONORIFICS = {
    "mr", "mr.", "mister", "mrs", "mrs.", "mistress", "ms", "miss", "ms.",
    "sir", "madam", "lady", "lord", "dr", "dr.", "doctor", "prof", "prof.",
    "professor", "captain", "capt", "capt.", "commander", "major", "colonel", 
    "col", "col.", "general", "gen.", "sergeant", "sgt", "sgt.", "lt", "lt.", 
    "lieutenant", "judge", "justice", "attorney", "barrister", "esquire", 
    "honorable", "hon.", "rev.", "reverend", "esq.", "esq"
}

def add_tokenizer_exceptions(nlp):
    """Add special tokenization rules for honorifics."""
    for honorific in HONORIFICS:
        for variant in {honorific, honorific.capitalize(), honorific.upper()}:
            nlp.tokenizer.add_special_case(variant, [{ORTH: variant}])

def calculate_pmi(cooccurrence_count, term1_count, term2_count, total_terms):
    """Calculate Pointwise Mutual Information for term pairs."""
    if cooccurrence_count == 0:
        return 0
    
    # P(term1, term2)
    p_xy = cooccurrence_count / total_terms
    # P(term1) * P(term2)
    p_x_p_y = (term1_count / total_terms) * (term2_count / total_terms)
    
    if p_x_p_y == 0:
        return 0
    
    return log(p_xy / p_x_p_y)

def extract_bigrams_from_sentence(doc_sentence, valid_terms_set):
    """Extract bigrams from a sentence where both terms are in our valid set."""
    bigrams = []
    tokens = [token.text.lower() for token in doc_sentence if not token.is_punct and not token.is_space]
    
    # Generate all possible bigrams in the sentence
    for i in range(len(tokens) - 1):
        term1 = tokens[i]
        term2 = tokens[i + 1]
        
        # Only include if both terms are in our extracted terms
        if term1 in valid_terms_set and term2 in valid_terms_set and term1 != term2:
            # Sort to ensure consistent ordering
            bigram = tuple(sorted([term1, term2]))
            bigrams.append(bigram)
    
    return bigrams

def extract_terms_with_context(chunks, nlp_model, progress_callback=None, debug_mode=False):
    """Extract terms using a simplified, fast approach with bigram co-occurrence tracking."""
    term_counter = Counter()
    term_contexts = defaultdict(list)
    bigram_cooccurrence = Counter()
    sentence_cooccurrence = Counter()  # Track terms that appear in same sentence
    
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        try:
            doc = nlp_model(chunk)
            
            # Track what we've already found to avoid duplicates
            found_in_chunk = set()
            
            # Method 1: Regex for honorific + name patterns (FIRST to catch complete names)
            honorific_pattern = r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sir|Lord|Lady)\s+([A-Z][a-zA-Z\']+(?:\s+[A-Z][a-zA-Z\']+)*)'
            honorific_matches = list(re.finditer(honorific_pattern, chunk))
            
            if debug_mode and honorific_matches:
                print(f"Chunk {i}: Found {len(honorific_matches)} honorific matches:")
                for match in honorific_matches:
                    print(f"  - '{match.group(0)}'")
            
            for match in honorific_matches:
                term = match.group(0).strip()
                term_lower = term.lower()
                if term_lower not in found_in_chunk:
                    found_in_chunk.add(term_lower)
                    # Also add just the name part to prevent separate extraction
                    name_part = match.group(2).lower()
                    found_in_chunk.add(name_part)
                    
                    term_counter[term] += 1
                    # Find the sentence containing this match
                    for sent in doc.sents:
                        if term in sent.text:  # Exact case match for better accuracy
                            context = sent.text.strip()
                            if context not in term_contexts[term]:
                                term_contexts[term].append(context)
                            break
            
            # Method 2: Named entities (but skip if already found with honorific)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT"}:
                    term = ent.text.strip()
                    if len(term) > 1:
                        term_lower = term.lower()
                        # Check if this entity overlaps with any honorific matches
                        is_part_of_honorific = False
                        for match in honorific_matches:
                            honorific_term = match.group(0).strip()
                            if term.lower() in honorific_term.lower():
                                is_part_of_honorific = True
                                break
                        
                        if term_lower not in found_in_chunk and not is_part_of_honorific:
                            found_in_chunk.add(term_lower)
                            term_counter[term] += 1
                            context = ent.sent.text.strip()
                            if context not in term_contexts[term]:
                                term_contexts[term].append(context)
            
            # Method 3: Noun phrases
            for chunk_span in doc.noun_chunks:
                original = chunk_span.text.strip()
                cleaned = clean_term(original)
                
                # Only accept if cleaned term is meaningful
                if (len(cleaned) > 2 and 
                    len(cleaned.split()) >= 1 and
                    not cleaned.lower() in STOP_WORDS):
                    
                    term = cleaned
                    term_lower = term.lower()
                    if term_lower not in found_in_chunk:
                        found_in_chunk.add(term_lower)
                        term_counter[term] += 1
                        context = chunk_span.sent.text.strip()
                        if context not in term_contexts[term]:
                            term_contexts[term].append(context)

            # Extract bigram co-occurrences from each sentence
            for sent in doc.sents:
                # Get terms that appear in this sentence
                sent_terms = set()
                sent_text_lower = sent.text.lower()
                
                # Find which of our extracted terms appear in this sentence
                for term in found_in_chunk:
                    if term in sent_text_lower:
                        sent_terms.add(term)
                
                # Record sentence-level co-occurrences
                if len(sent_terms) > 1:
                    for term1, term2 in itertools.combinations(sorted(sent_terms), 2):
                        sentence_cooccurrence[(term1, term2)] += 1
                
                # Extract adjacent bigrams within the sentence
                bigrams = extract_bigrams_from_sentence(sent, found_in_chunk)
                for bigram in bigrams:
                    bigram_cooccurrence[bigram] += 1

            if progress_callback:
                progress_callback(i + 1, total_chunks)
                
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            continue

    # Simple post-processing: remove very short or low-quality terms
    terms_to_remove = set()
    for term in term_counter:
        # Remove single characters or very short terms
        if len(term.strip()) < 2:
            terms_to_remove.add(term)
        # Remove pure stopwords (but preserve honorific combinations)
        elif (term.lower() in STOP_WORDS and 
              not any(word.lower().rstrip('.') in HONORIFICS for word in term.split())):
            terms_to_remove.add(term)
        # Remove terms that are just articles
        elif term.lower() in {'the', 'a', 'an'}:
            terms_to_remove.add(term)
    
    for term in terms_to_remove:
        term_counter.pop(term, None)
        term_contexts.pop(term, None)

    # Simple redundancy removal
    final_terms = dict(term_counter)
    terms_list = list(final_terms.keys())
    
    for i, term1 in enumerate(terms_list):
        if term1 not in final_terms:
            continue
        for j, term2 in enumerate(terms_list):
            if (i != j and term2 in final_terms and 
                term1.lower() in term2.lower() and 
                len(term1.split()) < len(term2.split()) and
                final_terms[term1] <= final_terms[term2]):
                final_terms.pop(term1, None)
                break

    # Filter contexts to match final terms
    filtered_contexts = {term: term_contexts[term] for term in final_terms if term in term_contexts}

    # Clean up co-occurrence data to only include final terms
    final_term_set = {term.lower() for term in final_terms.keys()}
    
    cleaned_bigram_cooccurrence = {}
    for (term1, term2), count in bigram_cooccurrence.items():
        if term1 in final_term_set and term2 in final_term_set:
            cleaned_bigram_cooccurrence[(term1, term2)] = count
    
    cleaned_sentence_cooccurrence = {}
    for (term1, term2), count in sentence_cooccurrence.items():
        if term1 in final_term_set and term2 in final_term_set:
            cleaned_sentence_cooccurrence[(term1, term2)] = count

    return final_terms, filtered_contexts, cleaned_bigram_cooccurrence, cleaned_sentence_cooccurrence

def extract_text_from_file(file_path):
    """Extract text from various file formats."""
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".txt":
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
    elif ext == ".xliff":
        tree = ET.parse(file_path)
        root = tree.getroot()
        # Handle namespace
        ns = {"ns": root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
        texts = []
        for source in root.findall(".//source" if not ns else ".//ns:source", ns):
            if source.text:
                texts.append(source.text)
        return "\n\n".join(texts)
    elif ext in [".html", ".htm"]:
        with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
            soup = BeautifulSoup(f, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# NEW UTILITY FUNCTIONS FOR TERMBASE HANDLING

def load_termbase_from_csv(file_path):
    """Load existing termbase from CSV file."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Expected columns: Source Term, Target Term, Frequency, Primary Context, Notes, etc.
        termbase_data = []
        
        for _, row in df.iterrows():
            term_entry = {
                "term": str(row.get("Source Term", row.get("source_term", ""))).strip(),
                "target_term": str(row.get("Target Term", row.get("target_term", ""))).strip(),
                "freq": int(row.get("Frequency", row.get("freq", 1))),
                "contexts": [str(row.get("Primary Context", row.get("context", "")))],
                "selected": True,
                "source": "imported-termbase",
                "validated": bool(row.get("Validated", row.get("validated", False))),
                "notes": str(row.get("Notes", row.get("notes", ""))).strip()
            }
            
            # Only add if we have a valid term
            if term_entry["term"]:
                termbase_data.append(term_entry)
        
        return termbase_data
    
    except Exception as e:
        raise ValueError(f"Failed to load termbase: {str(e)}")

def merge_termbases(existing_terms, new_terms):
    """Merge two termbases, handling duplicates intelligently."""
    merged_terms = {}
    
    # Add existing terms first
    for term_data in existing_terms:
        key = term_data["term"].lower()
        merged_terms[key] = term_data.copy()
        merged_terms[key]["is_existing"] = True
    
    # Add new terms, merging with existing ones
    for term_data in new_terms:
        key = term_data["term"].lower()
        
        if key in merged_terms:
            # Merge with existing term
            existing = merged_terms[key]
            
            # Update frequency (add them)
            existing["freq"] += term_data["freq"]
            
            # Merge contexts (avoid duplicates)
            existing_contexts = set(existing["contexts"])
            for context in term_data["contexts"]:
                if context not in existing_contexts:
                    existing["contexts"].append(context)
            
            # Update notes
            if term_data.get("notes"):
                if existing.get("notes"):
                    existing["notes"] += f"; {term_data['notes']}"
                else:
                    existing["notes"] = term_data["notes"]
            
            # Mark as updated
            existing["source"] = "merged"
            existing["is_updated"] = True
        
        else:
            # Add as new term
            new_term = term_data.copy()
            new_term["is_existing"] = False
            new_term["is_new"] = True
            merged_terms[key] = new_term
    
    return list(merged_terms.values())

def validate_termbase_structure(data):
    """Validate that termbase data has required structure."""
    required_fields = ["term", "freq", "contexts"]
    
    if not isinstance(data, list):
        raise ValueError("Termbase data must be a list of term entries")
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} must be a dictionary")
        
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Entry {i} missing required field: {field}")
    
    return True

# -------------------- Dialog Classes --------------------

class BulkEditDialog:
    def __init__(self, parent, terms, callback=None):
        self.terms = [t for t in terms if t.get("selected", True)]
        self.callback = callback
        
        if not self.terms:
            messagebox.showwarning("Bulk Edit", "No terms selected for bulk editing.")
            return
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Bulk Edit Terms")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Info label
        info_label = tk.Label(self.dialog, text=f"Bulk editing {len(self.terms)} selected terms")
        info_label.pack(pady=10)
        
        # Actions frame
        actions_frame = tk.LabelFrame(self.dialog, text="Bulk Actions")
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Validation actions
        validation_frame = tk.Frame(actions_frame)
        validation_frame.pack(fill=tk.X, pady=5)
        tk.Label(validation_frame, text="Validation:").pack(side=tk.LEFT)
        tk.Button(validation_frame, text="Mark All as Validated", 
                 command=self.validate_all).pack(side=tk.LEFT, padx=5)
        tk.Button(validation_frame, text="Mark All as Pending", 
                 command=self.unvalidate_all).pack(side=tk.LEFT, padx=5)
        
        # Source update
        source_frame = tk.Frame(actions_frame)
        source_frame.pack(fill=tk.X, pady=5)
        tk.Label(source_frame, text="Update Source:").pack(side=tk.LEFT)
        self.source_entry = tk.Entry(source_frame, width=20)
        self.source_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(source_frame, text="Apply", command=self.update_source).pack(side=tk.LEFT, padx=5)
        
        # Notes update
        notes_frame = tk.Frame(actions_frame)
        notes_frame.pack(fill=tk.X, pady=5)
        tk.Label(notes_frame, text="Append Notes:").pack(side=tk.LEFT)
        self.notes_entry = tk.Entry(notes_frame, width=30)
        self.notes_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(notes_frame, text="Apply", command=self.append_notes).pack(side=tk.LEFT, padx=5)
        
        # Pattern replacement
        pattern_frame = tk.LabelFrame(self.dialog, text="Pattern Replacement")
        pattern_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(pattern_frame, text="Find:").pack()
        self.find_entry = tk.Entry(pattern_frame, width=40)
        self.find_entry.pack(pady=2)
        
        tk.Label(pattern_frame, text="Replace with:").pack()
        self.replace_entry = tk.Entry(pattern_frame, width=40)
        self.replace_entry.pack(pady=2)
        
        replace_options_frame = tk.Frame(pattern_frame)
        replace_options_frame.pack(pady=5)
        
        self.case_sensitive_replace = tk.BooleanVar()
        tk.Checkbutton(replace_options_frame, text="Case sensitive", 
                      variable=self.case_sensitive_replace).pack(side=tk.LEFT)
        
        self.regex_replace = tk.BooleanVar()
        tk.Checkbutton(replace_options_frame, text="Use regex", 
                      variable=self.regex_replace).pack(side=tk.LEFT, padx=10)
        
        tk.Button(pattern_frame, text="Replace in Terms", 
                 command=self.replace_in_terms).pack(pady=5)
        
        # Close button
        tk.Button(self.dialog, text="Close", command=self.close_dialog).pack(pady=20)
    
    def validate_all(self):
        for term in self.terms:
            term["validated"] = True
        messagebox.showinfo("Success", f"Marked {len(self.terms)} terms as validated.")
        if self.callback:
            self.callback()
    
    def unvalidate_all(self):
        for term in self.terms:
            term["validated"] = False
        messagebox.showinfo("Success", f"Marked {len(self.terms)} terms as pending.")
        if self.callback:
            self.callback()
    
    def update_source(self):
        new_source = self.source_entry.get().strip()
        if not new_source:
            return
        
        for term in self.terms:
            term["source"] = new_source
        
        messagebox.showinfo("Success", f"Updated source for {len(self.terms)} terms.")
        if self.callback:
            self.callback()
    
    def append_notes(self):
        additional_notes = self.notes_entry.get().strip()
        if not additional_notes:
            return
        
        for term in self.terms:
            existing_notes = term.get("notes", "")
            if existing_notes:
                term["notes"] = existing_notes + "; " + additional_notes
            else:
                term["notes"] = additional_notes
        
        messagebox.showinfo("Success", f"Updated notes for {len(self.terms)} terms.")
        if self.callback:
            self.callback()
    
    def replace_in_terms(self):
        find_text = self.find_entry.get()
        replace_text = self.replace_entry.get()
        
        if not find_text:
            messagebox.showwarning("Replace", "Please enter text to find.")
            return
        
        replaced_count = 0
        
        try:
            for term in self.terms:
                original_term = term["term"]
                
                if self.regex_replace.get():
                    flags = 0 if self.case_sensitive_replace.get() else re.IGNORECASE
                    new_term = re.sub(find_text, replace_text, original_term, flags=flags)
                else:
                    if self.case_sensitive_replace.get():
                        new_term = original_term.replace(find_text, replace_text)
                    else:
                        # Case insensitive replacement
                        def replace_func(match):
                            return replace_text
                        new_term = re.sub(re.escape(find_text), replace_func, original_term, flags=re.IGNORECASE)
                
                if new_term != original_term:
                    term["term"] = new_term
                    replaced_count += 1
            
            messagebox.showinfo("Success", f"Replaced text in {replaced_count} terms.")
            if self.callback:
                self.callback()
                
        except re.error as e:
            messagebox.showerror("Regex Error", f"Invalid regex pattern: {str(e)}")
    
    def close_dialog(self):
        self.dialog.destroy()


class TermEditDialog:
    def __init__(self, parent, term_data, callback=None):
        self.term_data = term_data
        self.callback = callback
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Edit Term: {term_data['term']}")
        self.dialog.geometry("700x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Term details frame
        details_frame = tk.LabelFrame(self.dialog, text="Term Details")
        details_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Source term (read-only)
        tk.Label(details_frame, text="Source Term:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        source_label = tk.Label(details_frame, text=self.term_data["term"], font=("Arial", 10, "bold"))
        source_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Target term
        tk.Label(details_frame, text="Target Term:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.target_entry = tk.Entry(details_frame, width=40)
        self.target_entry.insert(0, self.term_data.get("target_term", ""))
        self.target_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Frequency (read-only)
        tk.Label(details_frame, text="Frequency:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        freq_label = tk.Label(details_frame, text=str(self.term_data["freq"]))
        freq_label.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Source
        tk.Label(details_frame, text="Source:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.source_entry = tk.Entry(details_frame, width=40)
        self.source_entry.insert(0, self.term_data.get("source", ""))
        self.source_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        # Validation status
        tk.Label(details_frame, text="Validated:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.validated_var = tk.BooleanVar(value=self.term_data.get("validated", False))
        tk.Checkbutton(details_frame, variable=self.validated_var).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        details_frame.columnconfigure(1, weight=1)
        
        # Notes frame
        notes_frame = tk.LabelFrame(self.dialog, text="Notes")
        notes_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.notes_text = tk.Text(notes_frame, height=4, wrap=tk.WORD)
        notes_scrollbar = ttk.Scrollbar(notes_frame, command=self.notes_text.yview)
        self.notes_text.configure(yscrollcommand=notes_scrollbar.set)
        self.notes_text.insert("1.0", self.term_data.get("notes", ""))
        
        self.notes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        notes_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Contexts frame
        contexts_frame = tk.LabelFrame(self.dialog, text="Contexts")
        contexts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Context list
        context_list_frame = tk.Frame(contexts_frame)
        context_list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.context_listbox = tk.Listbox(context_list_frame, height=6)
        context_listbox_scrollbar = ttk.Scrollbar(context_list_frame, command=self.context_listbox.yview)
        self.context_listbox.configure(yscrollcommand=context_listbox_scrollbar.set)
        
        # Populate contexts
        for i, context in enumerate(self.term_data.get("contexts", [])):
            display_context = (context[:80] + "...") if len(context) > 80 else context
            self.context_listbox.insert(tk.END, f"{i+1}. {display_context}")
        
        self.context_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        context_listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Bind selection event
        self.context_listbox.bind("<<ListboxSelect>>", self.on_context_selected)
        
        # Selected context display
        selected_context_frame = tk.LabelFrame(self.dialog, text="Selected Context")
        selected_context_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.selected_context_text = tk.Text(selected_context_frame, height=3, wrap=tk.WORD, bg="lightyellow")
        selected_context_scrollbar = ttk.Scrollbar(selected_context_frame, command=self.selected_context_text.yview)
        self.selected_context_text.configure(yscrollcommand=selected_context_scrollbar.set)
        
        self.selected_context_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        selected_context_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="Save Changes", command=self.save_changes, 
                 bg="lightgreen").pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def on_context_selected(self, event):
        """Handle context selection."""
        selection = self.context_listbox.curselection()
        if not selection or not self.term_data.get("contexts"):
            return
        
        idx = selection[0]
        contexts = self.term_data["contexts"]
        if 0 <= idx < len(contexts):
            context = contexts[idx]
            
            # Show context with highlighting
            self.selected_context_text.config(state=tk.NORMAL)
            self.selected_context_text.delete("1.0", tk.END)
            
            # Configure highlight tag
            self.selected_context_text.tag_config("highlight", background="yellow", foreground="black")
            
            # Insert context with term highlighting
            term = self.term_data["term"]
            last_end = 0
            for match in re.finditer(re.escape(term), context, re.IGNORECASE):
                # Insert text before match
                self.selected_context_text.insert(tk.END, context[last_end:match.start()])
                # Insert highlighted match
                self.selected_context_text.insert(tk.END, context[match.start():match.end()], "highlight")
                last_end = match.end()
            
            # Insert remaining text
            self.selected_context_text.insert(tk.END, context[last_end:])
            self.selected_context_text.config(state=tk.DISABLED)
    
    def save_changes(self):
        """Save changes to term data."""
        self.term_data["target_term"] = self.target_entry.get().strip()
        self.term_data["source"] = self.source_entry.get().strip()
        self.term_data["validated"] = self.validated_var.get()
        self.term_data["notes"] = self.notes_text.get("1.0", tk.END).strip()
        
        messagebox.showinfo("Saved", "Changes saved successfully.")
        
        if self.callback:
            self.callback()
        
        self.dialog.destroy()


class TermbaseImportDialog:
    """Dialog for importing existing termbase files."""
    
    def __init__(self, parent, callback=None):
        self.callback = callback
        self.imported_data = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Import Termbase")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # File selection frame
        file_frame = tk.LabelFrame(self.dialog, text="Select Termbase File")
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        file_controls = tk.Frame(file_frame)
        file_controls.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(file_controls, text="File:").pack(side=tk.LEFT)
        self.file_entry = tk.Entry(file_controls, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(file_controls, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        # Import options
        options_frame = tk.LabelFrame(self.dialog, text="Import Options")
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.merge_mode = tk.StringVar(value="merge")
        tk.Radiobutton(options_frame, text="Merge with existing terms (recommended)", 
                      variable=self.merge_mode, value="merge").pack(anchor="w", padx=5, pady=2)
        tk.Radiobutton(options_frame, text="Replace all existing terms", 
                      variable=self.merge_mode, value="replace").pack(anchor="w", padx=5, pady=2)
        tk.Radiobutton(options_frame, text="Import as separate termbase", 
                      variable=self.merge_mode, value="separate").pack(anchor="w", padx=5, pady=2)
        
        # Preview frame
        preview_frame = tk.LabelFrame(self.dialog, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Preview treeview
        preview_tree_frame = tk.Frame(preview_frame)
        preview_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        preview_columns = ("Source Term", "Target Term", "Frequency", "Validated", "Notes")
        self.preview_tree = ttk.Treeview(preview_tree_frame, columns=preview_columns, show="headings", height=8)
        
        for col in preview_columns:
            self.preview_tree.heading(col, text=col)
        
        self.preview_tree.column("Source Term", width=150)
        self.preview_tree.column("Target Term", width=150)
        self.preview_tree.column("Frequency", width=80, anchor="center")
        self.preview_tree.column("Validated", width=80, anchor="center")
        self.preview_tree.column("Notes", width=200)
        
        preview_scrollbar = ttk.Scrollbar(preview_tree_frame, orient=tk.VERTICAL, command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=preview_scrollbar.set)
        
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status label
        self.status_label = tk.Label(preview_frame, text="No file selected", fg="gray")
        self.status_label.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="Load Preview", command=self.load_preview, 
                 bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Import", command=self.import_termbase, 
                 bg="lightgreen").pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def browse_file(self):
        """Browse for termbase file."""
        file_path = filedialog.askopenfilename(
            title="Select Termbase File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
    
    def load_preview(self):
        """Load and preview the termbase file."""
        file_path = self.file_entry.get().strip()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid file.")
            return
        
        try:
            # Clear existing preview
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)
            
            # Load termbase data
            self.imported_data = load_termbase_from_csv(file_path)
            
            # Show preview (first 50 entries)
            preview_count = min(50, len(self.imported_data))
            for i in range(preview_count):
                term_data = self.imported_data[i]
                self.preview_tree.insert("", "end", values=(
                    term_data["term"],
                    term_data.get("target_term", ""),
                    term_data["freq"],
                    "Yes" if term_data.get("validated", False) else "No",
                    (term_data.get("notes", "")[:50] + "...") if len(term_data.get("notes", "")) > 50 else term_data.get("notes", "")
                ))
            
            # Update status
            total_terms = len(self.imported_data)
            self.status_label.config(
                text=f"Loaded {total_terms} terms (showing first {preview_count})", 
                fg="green"
            )
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to load termbase:\n{str(e)}")
            self.status_label.config(text="Error loading file", fg="red")
    
    def import_termbase(self):
        """Import the termbase with selected options."""
        if not self.imported_data:
            messagebox.showwarning("Import", "Please load a preview first.")
            return
        
        if self.callback:
            self.callback(self.imported_data, self.merge_mode.get())
        
        self.dialog.destroy()


class CorpusLoadDialog:
    """Dialog for loading corpus files separately from term extraction."""
    
    def __init__(self, parent, callback=None):
        self.callback = callback
        self.corpus_text = ""
        self.corpus_info = {}
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Load Corpus")
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # File selection
        file_frame = tk.LabelFrame(self.dialog, text="Select Corpus File")
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        file_controls = tk.Frame(file_frame)
        file_controls.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(file_controls, text="File:").pack(side=tk.LEFT)
        self.file_entry = tk.Entry(file_controls, width=40)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(file_controls, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        # File info display
        info_frame = tk.LabelFrame(self.dialog, text="Corpus Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, bg="lightyellow")
        info_scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="Analyze File", command=self.analyze_file, 
                 bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Load Corpus", command=self.load_corpus, 
                 bg="lightgreen").pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def browse_file(self):
        """Browse for corpus file."""
        file_path = filedialog.askopenfilename(
            title="Select Corpus File",
            filetypes=[
                ("All supported", "*.txt *.docx *.xliff *.html *.htm"),
                ("Text files", "*.txt"),
                ("Word documents", "*.docx"),
                ("XLIFF files", "*.xliff"),
                ("HTML files", "*.html *.htm"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
    
    def analyze_file(self):
        """Analyze the corpus file and show information."""
        file_path = self.file_entry.get().strip()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid file.")
            return
        
        try:
            # Extract text
            text = extract_text_from_file(file_path)
            
            # Calculate statistics
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.splitlines())
            paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Store for loading
            self.corpus_text = text
            self.corpus_info = {
                'file_path': file_path,
                'file_size': file_size,
                'file_type': file_ext,
                'char_count': char_count,
                'word_count': word_count,
                'line_count': line_count,
                'paragraph_count': paragraph_count
            }
            
            # Display information
            info_text = f"""File: {os.path.basename(file_path)}
File Type: {file_ext.upper()}
File Size: {file_size:,} bytes ({file_size/1024:.1f} KB)

Text Statistics:
- Characters: {char_count:,}
- Words: {word_count:,}
- Lines: {line_count:,}
- Paragraphs: {paragraph_count:,}

Estimated processing time: {self.estimate_processing_time(word_count)}

Preview (first 200 characters):
{text[:200]}{'...' if len(text) > 200 else ''}"""
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete("1.0", tk.END)
            self.info_text.insert("1.0", info_text)
            self.info_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze file:\n{str(e)}")
    
    def estimate_processing_time(self, word_count):
        """Estimate processing time based on word count."""
        if word_count < 1000:
            return "< 1 second"
        elif word_count < 10000:
            return "1-5 seconds"
        elif word_count < 50000:
            return "5-30 seconds"
        elif word_count < 100000:
            return "30-60 seconds"
        else:
            return f"1-{word_count//50000} minutes"
    
    def load_corpus(self):
        """Load the corpus."""
        if not self.corpus_text:
            messagebox.showwarning("Load Corpus", "Please analyze the file first.")
            return
        
        if self.callback:
            self.callback(self.corpus_text, self.corpus_info)
        
        self.dialog.destroy()

# -------------------- GUI Application - Part 1 --------------------

class TermExtractorApp:
    def __init__(self, root, nlp_model):
        self.root = root
        self.root.title("Term Base Creator")
        self.root.geometry("1400x900")
        self.nlp = nlp_model
        self.nlp.max_length = 2_000_000

        # Data storage
        self.term_data = []
        self.imported_termbase = []  # Store imported termbase separately
        self.bigram_cooccurrence = {}
        self.sentence_cooccurrence = {}
        self.current_term = None
        self.current_context_index = 0
        self.sort_reverse = {"Term": False, "Frequency": True, "Context": False}
        self.corpus_text = ""  # Store the full corpus for searching
        self.corpus_chunks = []  # Store processed chunks
        self.corpus_info = {}  # Store corpus metadata
        self.corpus_loaded = False

        self.create_widgets()

    def create_widgets(self):
        # Top-level corpus and termbase controls
        self.create_top_controls()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Terms tab
        self.terms_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.terms_frame, text="Auto-Extract")
        self.create_terms_tab()

        # Manual search tab
        self.search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.search_frame, text="Manual Search")
        self.create_search_tab()

        # Validation tab
        self.validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.validation_frame, text="Term Validation")
        self.create_validation_tab()

        # Co-occurrence tab
        self.cooccurrence_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.cooccurrence_frame, text="Co-occurrence")
        self.create_cooccurrence_tab()

    def create_top_controls(self):
        """Create top-level controls for corpus and termbase management."""
        top_frame = tk.Frame(self.root, bg="lightgray")
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Corpus controls
        corpus_frame = tk.LabelFrame(top_frame, text="Corpus Management", bg="lightgray")
        corpus_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        corpus_controls = tk.Frame(corpus_frame, bg="lightgray")
        corpus_controls.pack(fill=tk.X, padx=5, pady=5)
        
        self.corpus_label = tk.Label(corpus_controls, text="No corpus loaded", 
                                   fg="gray", bg="lightgray")
        self.corpus_label.pack(side=tk.LEFT)
        
        tk.Button(corpus_controls, text="Load Corpus", command=self.load_corpus_dialog,
                 bg="lightblue").pack(side=tk.RIGHT, padx=2)
        tk.Button(corpus_controls, text="Corpus Info", command=self.show_corpus_info,
                 state=tk.DISABLED).pack(side=tk.RIGHT, padx=2)
        
        # Termbase controls
        termbase_frame = tk.LabelFrame(top_frame, text="Termbase Management", bg="lightgray")
        termbase_frame.pack(side=tk.RIGHT, padx=5)
        
        termbase_controls = tk.Frame(termbase_frame, bg="lightgray")
        termbase_controls.pack(padx=5, pady=5)
        
        tk.Button(termbase_controls, text="Import Termbase", command=self.import_termbase_dialog,
                 bg="lightgreen").pack(side=tk.LEFT, padx=2)
        tk.Button(termbase_controls, text="Export Current", command=self.export_current_termbase,
                 bg="lightcoral").pack(side=tk.LEFT, padx=2)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_corpus_dialog(self):
        """Open corpus loading dialog."""
        def on_corpus_loaded(corpus_text, corpus_info):
            self.corpus_text = corpus_text
            self.corpus_info = corpus_info
            self.corpus_chunks = chunk_text(corpus_text)
            self.corpus_loaded = True
            
            # Update UI
            filename = os.path.basename(corpus_info['file_path'])
            word_count = corpus_info['word_count']
            self.corpus_label.config(
                text=f"Loaded: {filename} ({word_count:,} words)", 
                fg="green"
            )
            
            # Enable corpus info button
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.LabelFrame) and child.cget('text') == 'Corpus Management':
                            for btn in child.winfo_children()[0].winfo_children():
                                if isinstance(btn, tk.Button) and btn.cget('text') == 'Corpus Info':
                                    btn.config(state=tk.NORMAL)
            
            self.status_bar.config(text=f"Corpus loaded: {word_count:,} words, {len(self.corpus_chunks)} chunks")
            messagebox.showinfo("Corpus Loaded", f"Successfully loaded corpus with {word_count:,} words.")
        
        CorpusLoadDialog(self.root, on_corpus_loaded)

    def show_corpus_info(self):
        """Show detailed corpus information."""
        if not self.corpus_loaded:
            messagebox.showwarning("No Corpus", "No corpus is currently loaded.")
            return
        
        info = self.corpus_info
        info_text = f"""Corpus Information:

File: {os.path.basename(info['file_path'])}
Full Path: {info['file_path']}
File Type: {info['file_type'].upper()}
File Size: {info['file_size']:,} bytes ({info['file_size']/1024:.1f} KB)

Text Statistics:
- Characters: {info['char_count']:,}
- Words: {info['word_count']:,}
- Lines: {info['line_count']:,}
- Paragraphs: {info['paragraph_count']:,}
- Processing Chunks: {len(self.corpus_chunks)}

Average words per chunk: {info['word_count'] // len(self.corpus_chunks) if self.corpus_chunks else 0:,}"""
        
        messagebox.showinfo("Corpus Information", info_text)

    def import_termbase_dialog(self):
        """Open termbase import dialog."""
        def on_termbase_imported(imported_data, merge_mode):
            if merge_mode == "replace":
                self.term_data = imported_data
                self.imported_termbase = []
                message = f"Replaced current termbase with {len(imported_data)} imported terms."
            
            elif merge_mode == "separate":
                self.imported_termbase = imported_data
                message = f"Imported {len(imported_data)} terms as separate termbase."
            
            else:  # merge
                merged_data = merge_termbases(self.term_data, imported_data)
                self.term_data = merged_data
                self.imported_termbase = []
                message = f"Merged termbase: {len(merged_data)} total terms."
            
            # Update all views
            self.update_treeview()
            self.update_validation_view()
            self.status_bar.config(text=message)
            messagebox.showinfo("Import Successful", message)
        
        TermbaseImportDialog(self.root, on_termbase_imported)

    def export_current_termbase(self):
        """Export current termbase to file."""
        if not self.term_data and not self.imported_termbase:
            messagebox.showwarning("Export", "No termbase data to export.")
            return
        
        # Combine all term data
        all_terms = self.term_data + self.imported_termbase
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Export Termbase"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare export data
            export_data = []
            for term_data in all_terms:
                primary_context = term_data["contexts"][0] if term_data["contexts"] else ""
                export_data.append({
                    "Source Term": term_data["term"],
                    "Target Term": term_data.get("target_term", ""),
                    "Frequency": term_data["freq"],
                    "Primary Context": primary_context,
                    "Total Contexts": len(term_data["contexts"]),
                    "Source": term_data.get("source", ""),
                    "Validated": term_data.get("validated", False),
                    "Notes": term_data.get("notes", "")
                })
            
            # Export based on file extension
            if file_path.endswith('.xlsx'):
                df = pd.DataFrame(export_data)
                df.to_excel(file_path, index=False)
            else:
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            messagebox.showinfo("Export Successful", 
                              f"Exported {len(all_terms)} terms to:\n{file_path}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export termbase:\n{str(e)}")

    def create_validation_tab(self):
        """Create the validation tab interface."""
        # Control frame
        control_frame = tk.Frame(self.validation_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Filter controls
        filter_frame = tk.LabelFrame(control_frame, text="Filters")
        filter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        filter_controls = tk.Frame(filter_frame)
        filter_controls.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(filter_controls, text="Show:").pack(side=tk.LEFT)
        
        self.validation_filter = tk.StringVar(value="all")
        tk.Radiobutton(filter_controls, text="All", variable=self.validation_filter, 
                      value="all", command=self.update_validation_view).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(filter_controls, text="Validated", variable=self.validation_filter, 
                      value="validated", command=self.update_validation_view).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(filter_controls, text="Pending", variable=self.validation_filter, 
                      value="pending", command=self.update_validation_view).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(filter_controls, text="New Terms", variable=self.validation_filter, 
                      value="new", command=self.update_validation_view).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(filter_controls, text="Imported", variable=self.validation_filter, 
                      value="imported", command=self.update_validation_view).pack(side=tk.LEFT, padx=5)
        
        # Action controls
        action_frame = tk.LabelFrame(control_frame, text="Actions")
        action_frame.pack(side=tk.RIGHT, padx=5)
        
        action_controls = tk.Frame(action_frame)
        action_controls.pack(padx=5, pady=5)
        
        tk.Button(action_controls, text="Bulk Edit", command=self.open_bulk_edit_dialog,
                 bg="lightyellow").pack(side=tk.LEFT, padx=2)
        tk.Button(action_controls, text="Export Selected", command=self.export_validated_terms,
                 bg="lightgreen").pack(side=tk.LEFT, padx=2)
        
        # Main validation treeview
        tree_frame = tk.Frame(self.validation_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview
        validation_columns = ("Selected", "Term", "Target Term", "Frequency", "Source", "Validated", "Notes")
        self.validation_tree = ttk.Treeview(tree_frame, columns=validation_columns, show="headings", height=15)
        
        # Configure headings
        self.validation_tree.heading("Selected", text="✓")
        self.validation_tree.heading("Term", text="Source Term")
        self.validation_tree.heading("Target Term", text="Target Term")
        self.validation_tree.heading("Frequency", text="Freq")
        self.validation_tree.heading("Source", text="Source")
        self.validation_tree.heading("Validated", text="Validated")
        self.validation_tree.heading("Notes", text="Notes")
        
        # Configure column widths
        self.validation_tree.column("Selected", width=50, anchor="center")
        self.validation_tree.column("Term", width=200)
        self.validation_tree.column("Target Term", width=200)
        self.validation_tree.column("Frequency", width=60, anchor="center")
        self.validation_tree.column("Source", width=100, anchor="center")
        self.validation_tree.column("Validated", width=70, anchor="center")
        self.validation_tree.column("Notes", width=250)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.validation_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.validation_tree.xview)
        self.validation_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.validation_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind events
        self.validation_tree.bind("<Double-1>", self.on_validation_term_double_click)
        self.validation_tree.bind("<Button-1>", self.handle_validation_checkbox_click)
        
        # Term details frame for validation
        details_frame = tk.LabelFrame(self.validation_frame, text="Term Details")
        details_frame.pack(fill=tk.X, padx=10, pady=5)
        
        details_content = tk.Frame(details_frame)
        details_content.pack(fill=tk.X, padx=5, pady=5)
        
        # Quick edit controls
        tk.Label(details_content, text="Target Term:").pack(side=tk.LEFT)
        self.quick_target_entry = tk.Entry(details_content, width=30)
        self.quick_target_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(details_content, text="Notes:").pack(side=tk.LEFT, padx=(20,5))
        self.quick_notes_entry = tk.Entry(details_content, width=40)
        self.quick_notes_entry.pack(side=tk.LEFT, padx=5)
        
        self.quick_validated_var = tk.BooleanVar()
        tk.Checkbutton(details_content, text="Validated", 
                      variable=self.quick_validated_var).pack(side=tk.LEFT, padx=10)
        
        tk.Button(details_content, text="Save Changes", command=self.save_quick_edits,
                 bg="lightgreen").pack(side=tk.RIGHT, padx=5)

    def update_validation_view(self):
        """Update the validation tab view based on current filters."""
        # Clear existing items
        for item in self.validation_tree.get_children():
            self.validation_tree.delete(item)
        
        # Get filter value
        filter_value = self.validation_filter.get()
        
        # Combine all terms
        all_terms = []
        
        # Add extracted terms
        for term_data in self.term_data:
            term_copy = term_data.copy()
            term_copy["term_type"] = "extracted"
            all_terms.append(term_copy)
        
        # Add imported terms
        for term_data in self.imported_termbase:
            term_copy = term_data.copy()
            term_copy["term_type"] = "imported"
            all_terms.append(term_copy)
        
        # Apply filter
        filtered_terms = []
        for term_data in all_terms:
            if filter_value == "all":
                filtered_terms.append(term_data)
            elif filter_value == "validated" and term_data.get("validated", False):
                filtered_terms.append(term_data)
            elif filter_value == "pending" and not term_data.get("validated", False):
                filtered_terms.append(term_data)
            elif filter_value == "new" and term_data.get("is_new", False):
                filtered_terms.append(term_data)
            elif filter_value == "imported" and term_data.get("term_type") == "imported":
                filtered_terms.append(term_data)
        
        # Populate treeview
        for i, term_data in enumerate(filtered_terms):
            # Color coding based on term status
            tags = []
            if term_data.get("validated", False):
                tags.append("validated")
            elif term_data.get("is_new", False):
                tags.append("new")
            elif term_data.get("term_type") == "imported":
                tags.append("imported")
            
            self.validation_tree.insert("", "end", iid=str(i), tags=tags, values=(
                "✓" if term_data.get("selected", True) else "",
                term_data["term"],
                term_data.get("target_term", ""),
                term_data["freq"],
                term_data.get("source", ""),
                "Yes" if term_data.get("validated", False) else "No",
                term_data.get("notes", "")[:50] + ("..." if len(term_data.get("notes", "")) > 50 else "")
            ))
        
        # Configure tags for visual distinction
        self.validation_tree.tag_configure("validated", background="lightgreen")
        self.validation_tree.tag_configure("new", background="lightyellow")
        self.validation_tree.tag_configure("imported", background="lightblue")
        
        # Store filtered terms for reference
        self.filtered_validation_terms = filtered_terms

    def handle_validation_checkbox_click(self, event):
        """Handle clicking on checkbox in validation view."""
        if self.validation_tree.identify_column(event.x) != "#1":  # Not the checkbox column
            return
        
        row_id = self.validation_tree.identify_row(event.y)
        if not row_id:
            return
        
        idx = int(row_id)
        if 0 <= idx < len(getattr(self, 'filtered_validation_terms', [])):
            # Toggle selection
            term_data = self.filtered_validation_terms[idx]
            term_data["selected"] = not term_data.get("selected", True)
            self.update_validation_view()

    def on_validation_term_double_click(self, event):
        """Handle double-click on term in validation view."""
        selection = self.validation_tree.selection()
        if not selection:
            return
        
        idx = int(selection[0])
        if 0 <= idx < len(getattr(self, 'filtered_validation_terms', [])):
            term_data = self.filtered_validation_terms[idx]
            
            # Open term edit dialog
            def refresh_callback():
                self.update_validation_view()
                self.update_treeview()
            
            TermEditDialog(self.root, term_data, refresh_callback)

    def save_quick_edits(self):
        """Save quick edits from validation tab."""
        selection = self.validation_tree.selection()
        if not selection:
            messagebox.showwarning("Save", "Please select a term to edit.")
            return
        
        idx = int(selection[0])
        if 0 <= idx < len(getattr(self, 'filtered_validation_terms', [])):
            term_data = self.filtered_validation_terms[idx]
            
            # Update term data
            term_data["target_term"] = self.quick_target_entry.get().strip()
            term_data["notes"] = self.quick_notes_entry.get().strip()
            term_data["validated"] = self.quick_validated_var.get()
            
            # Refresh views
            self.update_validation_view()
            self.update_treeview()
            
            # Clear quick edit fields
            self.quick_target_entry.delete(0, tk.END)
            self.quick_notes_entry.delete(0, tk.END)
            self.quick_validated_var.set(False)
            
            messagebox.showinfo("Saved", "Term updated successfully.")

    def open_bulk_edit_dialog(self):
        """Open bulk edit dialog for selected terms."""
        if not hasattr(self, 'filtered_validation_terms'):
            messagebox.showwarning("Bulk Edit", "No terms available for editing.")
            return
        
        selected_terms = [t for t in self.filtered_validation_terms if t.get("selected", True)]
        
        def refresh_callback():
            self.update_validation_view()
            self.update_treeview()
        
        BulkEditDialog(self.root, selected_terms, refresh_callback)

    def export_validated_terms(self):
        """Export selected validated terms."""
        if not hasattr(self, 'filtered_validation_terms'):
            messagebox.showwarning("Export", "No terms available for export.")
            return
        
        selected_terms = [t for t in self.filtered_validation_terms if t.get("selected", True)]
        
        if not selected_terms:
            messagebox.showwarning("Export", "No terms selected for export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="Export Validated Terms"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare export data
            export_data = []
            for term_data in selected_terms:
                primary_context = term_data["contexts"][0] if term_data.get("contexts") else ""
                export_data.append({
                    "Source Term": term_data["term"],
                    "Target Term": term_data.get("target_term", ""),
                    "Frequency": term_data["freq"],
                    "Primary Context": primary_context,
                    "Total Contexts": len(term_data.get("contexts", [])),
                    "Source": term_data.get("source", ""),
                    "Validated": term_data.get("validated", False),
                    "Notes": term_data.get("notes", ""),
                    "Term Type": term_data.get("term_type", "extracted")
                })
            
            # Export based on file extension
            if file_path.endswith('.xlsx'):
                df = pd.DataFrame(export_data)
                df.to_excel(file_path, index=False)
            else:
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            messagebox.showinfo("Export Successful", 
                              f"Exported {len(selected_terms)} validated terms to:\n{file_path}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export terms:\n{str(e)}")

# -------------------- GUI Application - Part 2 --------------------

    def create_terms_tab(self):
        # Extraction options frame (simplified since corpus is loaded at top level)
        options_frame = tk.Frame(self.terms_frame)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(options_frame, text="Min frequency:").pack(side=tk.LEFT)
        self.min_freq_entry = tk.Entry(options_frame, width=5)
        self.min_freq_entry.insert(0, "2")
        self.min_freq_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(options_frame, text="Extract Terms", 
                 command=self.extract_terms_thread, bg="lightblue").pack(side=tk.LEFT, padx=20)

        # Add debug button to see extraction details
        tk.Button(options_frame, text="Debug Mode", 
                 command=self.toggle_debug_mode, bg="lightyellow").pack(side=tk.LEFT, padx=10)
        self.debug_mode = False

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.terms_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        # Status label
        self.status_label = tk.Label(self.terms_frame, text="Ready to extract terms", fg="blue")
        self.status_label.pack()

        # Term count label
        self.term_count_label = tk.Label(self.terms_frame, text="(0 terms extracted)")
        self.term_count_label.pack()

        # Main treeview for terms
        tree_frame = tk.Frame(self.terms_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("Selected", "Term", "Frequency", "Context")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        
        # Configure column headings with sort functionality
        self.tree.heading("Selected", text="✓")
        self.tree.heading("Term", text="Term", command=lambda: self.sort_by_column("Term"))
        self.tree.heading("Frequency", text="Freq", command=lambda: self.sort_by_column("Frequency"))
        self.tree.heading("Context", text="Context", command=lambda: self.sort_by_column("Context"))

        # Configure column widths
        self.tree.column("Selected", width=50, anchor="center")
        self.tree.column("Term", width=250)
        self.tree.column("Frequency", width=80, anchor="center")
        self.tree.column("Context", width=500)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind events
        self.tree.bind("<<TreeviewSelect>>", self.on_term_selected)
        self.tree.bind("<Button-1>", self.handle_checkbox_click)
        self.tree.bind("<Double-1>", self.on_term_double_click)

        # Context display frame
        context_frame = tk.LabelFrame(self.terms_frame, text="Context Preview")
        context_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        self.context_text = tk.Text(context_frame, height=6, wrap=tk.WORD, bg="lightyellow")
        context_scrollbar = ttk.Scrollbar(context_frame, command=self.context_text.yview)
        self.context_text.configure(yscrollcommand=context_scrollbar.set)
        
        self.context_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        context_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Navigation buttons
        nav_frame = tk.Frame(self.terms_frame)
        nav_frame.pack(pady=5)

        self.prev_button = tk.Button(nav_frame, text="← Previous Context", 
                                   command=self.show_prev_context, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(nav_frame, text="Next Context →", 
                                   command=self.show_next_context, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Bottom frame for export
        bottom_frame = tk.Frame(self.terms_frame)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(bottom_frame, text="Select All", command=self.select_all_terms).pack(side=tk.LEFT)
        tk.Button(bottom_frame, text="Deselect All", command=self.deselect_all_terms).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Export Selected to CSV", 
                 command=self.export_selected, bg="lightgreen").pack(side=tk.RIGHT)

    def extract_terms_thread(self):
        """Run term extraction in a separate thread."""
        if not self.corpus_loaded:
            messagebox.showwarning("No Corpus", "Please load a corpus first using the 'Load Corpus' button.")
            return
        
        threading.Thread(target=self.extract_terms, daemon=True).start()

    def extract_terms(self):
        """Main term extraction logic with co-occurrence analysis."""
        try:
            min_freq = int(self.min_freq_entry.get())
        except ValueError:
            min_freq = 2
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Invalid frequency, using default: 2"))

        # Update UI
        self.root.after(0, lambda: self.status_label.config(text="Processing corpus..."))
        self.root.after(0, lambda: self.progress_var.set(0))

        try:
            # Progress callback
            def progress_callback(current, total):
                progress = (current / total) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Processing chunk {current}/{total}..."))

            # Extract terms with co-occurrence data
            term_counts, term_contexts, bigram_cooccur, sentence_cooccur = extract_terms_with_context(
                self.corpus_chunks, self.nlp, progress_callback, debug_mode=getattr(self, 'debug_mode', False))

            # Store co-occurrence data
            self.bigram_cooccurrence = bigram_cooccur
            self.sentence_cooccurrence = sentence_cooccur

            # Filter by minimum frequency and prepare term data
            extracted_terms = []
            for term, freq in term_counts.items():
                if freq >= min_freq:
                    contexts = term_contexts.get(term, [])
                    extracted_terms.append({
                        "term": term,
                        "freq": freq,
                        "contexts": contexts,
                        "selected": True,
                        "source": "auto-extracted",
                        "validated": False,
                        "target_term": "",
                        "notes": "",
                        "is_new": True
                    })

            # Sort by frequency (default)
            extracted_terms.sort(key=lambda x: x["freq"], reverse=True)
            
            # Add to existing terms (don't replace)
            self.term_data.extend(extracted_terms)

            # Update UI
            self.root.after(0, self.update_gui_after_extraction)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to process corpus:\n{str(e)}"))
            self.root.after(0, lambda: self.status_label.config(text="Error occurred"))

    def update_gui_after_extraction(self):
        """Update GUI after term extraction is complete."""
        self.update_treeview()
        self.update_cooccurrence_view()
        self.update_validation_view()
        extracted_count = len([t for t in self.term_data if t.get("is_new", False)])
        self.term_count_label.config(text=f"({extracted_count} new terms extracted)")
        self.status_label.config(text="Extraction complete", fg="green")
        self.progress_var.set(100)
        self.context_text.delete("1.0", tk.END)
        
        # Enable navigation buttons
        self.prev_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)

    def create_search_tab(self):
        """Create manual search interface."""
        # Search controls
        search_control_frame = tk.Frame(self.search_frame)
        search_control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(search_control_frame, text="Search Term:").pack(side=tk.LEFT)
        self.search_entry = tk.Entry(search_control_frame, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<Return>', lambda e: self.search_corpus())

        search_options_frame = tk.Frame(search_control_frame)
        search_options_frame.pack(side=tk.LEFT, padx=10)

        self.case_sensitive = tk.BooleanVar()
        tk.Checkbutton(search_options_frame, text="Case sensitive", 
                      variable=self.case_sensitive).pack(side=tk.LEFT)

        self.whole_word = tk.BooleanVar(value=True)
        tk.Checkbutton(search_options_frame, text="Whole word", 
                      variable=self.whole_word).pack(side=tk.LEFT, padx=5)

        tk.Button(search_control_frame, text="Search", command=self.search_corpus, 
                 bg="lightblue").pack(side=tk.LEFT, padx=10)
        tk.Button(search_control_frame, text="Add to Term Base", command=self.add_search_to_termbase,
                 bg="lightgreen", state=tk.DISABLED).pack(side=tk.LEFT, padx=5)

        # Search results
        results_frame = tk.LabelFrame(self.search_frame, text="Search Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Results info
        self.search_info_label = tk.Label(results_frame, text="No search performed", fg="gray")
        self.search_info_label.pack(pady=5)

        # Results treeview
        search_tree_frame = tk.Frame(results_frame)
        search_tree_frame.pack(fill=tk.BOTH, expand=True)

        search_columns = ("Context", "Position")
        self.search_tree = ttk.Treeview(search_tree_frame, columns=search_columns, show="headings", height=12)
        
        self.search_tree.heading("Context", text="Context")
        self.search_tree.heading("Position", text="Position in Text")
        
        self.search_tree.column("Context", width=600)
        self.search_tree.column("Position", width=120, anchor="center")

        search_scrollbar = ttk.Scrollbar(search_tree_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=search_scrollbar.set)

        self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        search_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Context preview
        preview_frame = tk.LabelFrame(self.search_frame, text="Context Preview")
        preview_frame.pack(fill=tk.X, padx=10, pady=5)

        self.search_context_text = tk.Text(preview_frame, height=4, wrap=tk.WORD, bg="lightyellow")
        search_context_scrollbar = ttk.Scrollbar(preview_frame, command=self.search_context_text.yview)
        self.search_context_text.configure(yscrollcommand=search_context_scrollbar.set)
        
        self.search_context_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        search_context_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind selection event
        self.search_tree.bind("<<TreeviewSelect>>", self.on_search_result_selected)

        # Variables to store search results
        self.current_search_term = ""
        self.current_search_results = []

    def create_cooccurrence_tab(self):
        """Create co-occurrence analysis interface."""
        # Control frame
        control_frame = tk.Frame(self.cooccurrence_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame, text="Analysis Type:").pack(side=tk.LEFT)
        self.analysis_type = tk.StringVar(value="bigram")
        tk.Radiobutton(control_frame, text="Adjacent Bigrams", variable=self.analysis_type, 
                      value="bigram", command=self.update_cooccurrence_view).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(control_frame, text="Sentence Co-occurrence", variable=self.analysis_type, 
                      value="sentence", command=self.update_cooccurrence_view).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Min Co-occurrence:").pack(side=tk.LEFT, padx=(20,5))
        self.min_cooccur_entry = tk.Entry(control_frame, width=5)
        self.min_cooccur_entry.insert(0, "2")
        self.min_cooccur_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Update", command=self.update_cooccurrence_view).pack(side=tk.LEFT, padx=5)

        # Co-occurrence treeview
        cooccur_tree_frame = tk.Frame(self.cooccurrence_frame)
        cooccur_tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        cooccur_columns = ("Term 1", "Term 2", "Co-occurrence Count", "PMI Score")
        self.cooccur_tree = ttk.Treeview(cooccur_tree_frame, columns=cooccur_columns, show="headings")
        
        for col in cooccur_columns:
            self.cooccur_tree.heading(col, text=col)
        
        self.cooccur_tree.column("Term 1", width=200)
        self.cooccur_tree.column("Term 2", width=200)
        self.cooccur_tree.column("Co-occurrence Count", width=150, anchor="center")
        self.cooccur_tree.column("PMI Score", width=150, anchor="center")

        cooccur_scrollbar = ttk.Scrollbar(cooccur_tree_frame, orient=tk.VERTICAL, command=self.cooccur_tree.yview)
        self.cooccur_tree.configure(yscrollcommand=cooccur_scrollbar.set)

        self.cooccur_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cooccur_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Export co-occurrence button
        export_cooccur_frame = tk.Frame(self.cooccurrence_frame)
        export_cooccur_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(export_cooccur_frame, text="Export Co-occurrence Data", 
                 command=self.export_cooccurrence, bg="lightgreen").pack(side=tk.RIGHT)

# -------------------- GUI Application - Part 3 --------------------

    def search_corpus(self):
        """Search for a term in the corpus."""
        search_term = self.search_entry.get().strip()
        if not search_term:
            messagebox.showwarning("Search", "Please enter a search term.")
            return
        
        if not self.corpus_loaded:
            messagebox.showwarning("No Corpus", "Please load a corpus first.")
            return

        # Clear previous results
        for item in self.search_tree.get_children():
            self.search_tree.delete(item)

        # Prepare search pattern
        if self.whole_word.get():
            pattern = r'\b' + re.escape(search_term) + r'\b'
        else:
            pattern = re.escape(search_term)

        flags = 0 if self.case_sensitive.get() else re.IGNORECASE

        # Find all matches
        matches = list(re.finditer(pattern, self.corpus_text, flags))
        
        if not matches:
            self.search_info_label.config(text=f"No matches found for '{search_term}'", fg="red")
            self.current_search_results = []
            return

        # Extract contexts for each match
        contexts = []
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            
            # Get surrounding context (±100 characters)
            context_start = max(0, start_pos - 50)
            context_end = min(len(self.corpus_text), end_pos + 50)
            
            # Try to break at word boundaries
            if context_start > 0:
                space_pos = self.corpus_text.find(' ', context_start)
                if space_pos != -1 and space_pos < start_pos:
                    context_start = space_pos + 1
            
            if context_end < len(self.corpus_text):
                space_pos = self.corpus_text.rfind(' ', end_pos, context_end)
                if space_pos != -1:
                    context_end = space_pos

            context = self.corpus_text[context_start:context_end].strip()
            
            # Highlight the search term in context
            term_in_context = self.corpus_text[start_pos:end_pos]
            highlighted_context = context.replace(term_in_context, f"**{term_in_context}**")
            
            contexts.append({
                'context': context,
                'highlighted': highlighted_context,
                'position': start_pos,
                'match': term_in_context
            })

        # Store results
        self.current_search_term = search_term
        self.current_search_results = contexts

        # Update UI
        self.search_info_label.config(text=f"Found {len(matches)} matches for '{search_term}'", fg="green")
        
        for i, ctx in enumerate(contexts):
            # Truncate long contexts for display
            display_context = (ctx['highlighted'][:100] + "...") if len(ctx['highlighted']) > 100 else ctx['highlighted']
            self.search_tree.insert("", "end", values=(display_context, ctx['position']))

        # Enable add button
        for widget in self.search_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button) and child.cget('text') == 'Add to Term Base':
                        child.config(state=tk.NORMAL)
                        break

    def on_search_result_selected(self, event):
        """Handle selection of search result."""
        selection = self.search_tree.selection()
        if not selection or not self.current_search_results:
            return

        idx = self.search_tree.index(selection[0])
        if 0 <= idx < len(self.current_search_results):
            context_data = self.current_search_results[idx]
            
            # Show full context with highlighting
            self.search_context_text.config(state=tk.NORMAL)
            self.search_context_text.delete("1.0", tk.END)
            
            # Configure highlight tag
            self.search_context_text.tag_config("highlight", background="yellow", foreground="black")
            
            context = context_data['context']
            match_term = context_data['match']
            
            # Insert context with highlighting
            last_end = 0
            for match in re.finditer(re.escape(match_term), context, re.IGNORECASE):
                # Insert text before match
                self.search_context_text.insert(tk.END, context[last_end:match.start()])
                # Insert highlighted match
                self.search_context_text.insert(tk.END, context[match.start():match.end()], "highlight")
                last_end = match.end()
            
            # Insert remaining text
            self.search_context_text.insert(tk.END, context[last_end:])
            self.search_context_text.config(state=tk.DISABLED)

    def add_search_to_termbase(self):
        """Add current search term to term base."""
        if not self.current_search_term or not self.current_search_results:
            messagebox.showwarning("Add Term", "No search results to add.")
            return

        # Check if term already exists
        existing_term = next((t for t in self.term_data if t["term"].lower() == self.current_search_term.lower()), None)
        
        if existing_term:
            result = messagebox.askyesno("Term Exists", 
                                       f"'{self.current_search_term}' already exists in the term base.\n"
                                       "Do you want to update it with new search results?")
            if not result:
                return
            
            # Update existing term
            existing_contexts = existing_term["contexts"]
            new_contexts = [ctx['context'] for ctx in self.current_search_results]
            
            # Add new contexts that don't already exist
            for ctx in new_contexts:
                if ctx not in existing_contexts:
                    existing_contexts.append(ctx)
            
            existing_term["freq"] = len(self.current_search_results)
            existing_term["source"] = "manual-search (updated)"
            
            messagebox.showinfo("Updated", f"Updated '{self.current_search_term}' with {len(new_contexts)} contexts.")
        
        else:
            # Add new term
            new_term = {
                "term": self.current_search_term,
                "freq": len(self.current_search_results),
                "contexts": [ctx['context'] for ctx in self.current_search_results],
                "selected": True,
                "source": "manual-search",
                "validated": False,
                "target_term": "",
                "notes": "",
                "is_new": True
            }
            
            self.term_data.append(new_term)
            messagebox.showinfo("Added", f"Added '{self.current_search_term}' to term base with {len(self.current_search_results)} contexts.")

        # Refresh the terms view
        self.update_validation_view()
        self.update_treeview()

    def update_cooccurrence_view(self):
        """Update the co-occurrence analysis view."""
        # Clear existing items
        for item in self.cooccur_tree.get_children():
            self.cooccur_tree.delete(item)

        if not hasattr(self, 'bigram_cooccurrence') or not self.bigram_cooccurrence:
            return

        try:
            min_cooccur = int(self.min_cooccur_entry.get())
        except ValueError:
            min_cooccur = 2

        # Choose which co-occurrence data to display
        if self.analysis_type.get() == "bigram":
            cooccur_data = self.bigram_cooccurrence
        else:
            cooccur_data = self.sentence_cooccurrence

        # Calculate total terms for PMI
        total_terms = sum(item["freq"] for item in self.term_data)
        term_freq_map = {item["term"].lower(): item["freq"] for item in self.term_data}

        # Sort by co-occurrence count
        sorted_cooccur = sorted(cooccur_data.items(), key=lambda x: x[1], reverse=True)

        for (term1, term2), count in sorted_cooccur:
            if count >= min_cooccur:
                # Calculate PMI if we have frequency data
                pmi_score = 0
                if term1 in term_freq_map and term2 in term_freq_map:
                    pmi_score = calculate_pmi(count, term_freq_map[term1], 
                                            term_freq_map[term2], total_terms)
                
                self.cooccur_tree.insert("", "end", values=(
                    term1.title(),
                    term2.title(), 
                    count, 
                    f"{pmi_score:.3f}"
                ))

    def update_treeview(self):
        """Refresh the treeview with current term data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert updated data
        for i, item in enumerate(self.term_data):
            context = item["contexts"][0] if item["contexts"] else "No context available"
            # Truncate long contexts for display
            display_context = (context[:80] + "...") if len(context) > 80 else context
            
            self.tree.insert("", "end", iid=str(i), values=(
                "✓" if item["selected"] else "",
                item["term"],
                item["freq"],
                display_context
            ))

    def sort_by_column(self, column):
        """Sort terms by the specified column."""
        if column == "Term":
            self.sort_reverse[column] = not self.sort_reverse[column]
            self.term_data.sort(key=lambda x: x["term"], reverse=self.sort_reverse[column])
        elif column == "Frequency":
            self.sort_reverse[column] = not self.sort_reverse[column]
            self.term_data.sort(key=lambda x: x["freq"], reverse=self.sort_reverse[column])
        elif column == "Context":
            self.sort_reverse[column] = not self.sort_reverse[column]
            self.term_data.sort(key=lambda x: x["contexts"][0] if x["contexts"] else "", 
                              reverse=self.sort_reverse[column])
        
        self.update_treeview()

    def handle_checkbox_click(self, event):
        """Handle clicking on checkbox column."""
        if self.tree.identify_column(event.x) != "#1":  # Not the checkbox column
            return

        row_id = self.tree.identify_row(event.y)
        if not row_id:
            return

        idx = int(row_id)
        if 0 <= idx < len(self.term_data):
            # Toggle selection
            self.term_data[idx]["selected"] = not self.term_data[idx]["selected"]
            self.update_treeview()
            # Keep the row selected for context display
            self.tree.selection_set(row_id)

    def on_term_selected(self, event):
        """Handle term selection in treeview."""
        selection = self.tree.selection()
        if not selection:
            return

        idx = int(selection[0])
        if 0 <= idx < len(self.term_data):
            item = self.term_data[idx]
            self.current_term = item["term"]
            self.current_context_index = 0
            
            if item["contexts"]:
                self.show_context(self.current_term, item["contexts"][0])
            else:
                self.show_context(self.current_term, "No context available")

    def on_term_double_click(self, event):
        """Handle double-click on term (show edit dialog)."""
        selection = self.tree.selection()
        if not selection:
            return

        idx = int(selection[0])
        if 0 <= idx < len(self.term_data):
            item = self.term_data[idx]
            
            def refresh_callback():
                self.update_treeview()
                self.update_validation_view()
            
            TermEditDialog(self.root, item, refresh_callback)

    def show_context(self, term, context):
        """Display context with term highlighting."""
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        
        # Configure highlight tag
        self.context_text.tag_config("highlight", background="yellow", foreground="black")
        
        # Insert context with highlighting
        last_end = 0
        for match in re.finditer(re.escape(term), context, re.IGNORECASE):
            # Insert text before match
            self.context_text.insert(tk.END, context[last_end:match.start()])
            # Insert highlighted match
            self.context_text.insert(tk.END, context[match.start():match.end()], "highlight")
            last_end = match.end()
        
        # Insert remaining text
        self.context_text.insert(tk.END, context[last_end:])
        self.context_text.config(state=tk.DISABLED)

    def show_prev_context(self):
        """Show previous context for current term."""
        self._show_relative_context(-1)

    def show_next_context(self):
        """Show next context for current term."""
        self._show_relative_context(1)

    def _show_relative_context(self, direction):
        """Show context relative to current index."""
        if not self.current_term:
            return

        # Find current term in data
        idx = next((i for i, d in enumerate(self.term_data) if d["term"] == self.current_term), None)
        if idx is None:
            return

        contexts = self.term_data[idx]["contexts"]
        if not contexts:
            messagebox.showinfo("No Context", "No contexts available for this term.")
            return

        # Navigate to next/previous context
        self.current_context_index = (self.current_context_index + direction) % len(contexts)
        self.show_context(self.current_term, contexts[self.current_context_index])

    def select_all_terms(self):
        """Select all terms for export."""
        for item in self.term_data:
            item["selected"] = True
        self.update_treeview()

    def deselect_all_terms(self):
        """Deselect all terms."""
        for item in self.term_data:
            item["selected"] = False
        self.update_treeview()

    def export_selected(self):
        """Export selected terms to CSV."""
        selected_terms = [item for item in self.term_data if item["selected"]]
        
        if not selected_terms:
            messagebox.showwarning("Export", "No terms selected for export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save terms as CSV"
        )
        
        if not file_path:
            return

        try:
            # Prepare data for export
            export_data = []
            for item in selected_terms:
                primary_context = item["contexts"][0] if item["contexts"] else ""
                export_data.append({
                    "Source Term": item["term"],
                    "Target Term": item.get("target_term", ""),
                    "Frequency": item["freq"],
                    "Primary Context": primary_context,
                    "Total Contexts": len(item["contexts"]),
                    "Source": item.get("source", "auto-extracted"),
                    "Validated": item.get("validated", False),
                    "Notes": item.get("notes", "")
                })

            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            messagebox.showinfo("Export Successful", 
                              f"Exported {len(selected_terms)} terms to:\n{file_path}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file:\n{str(e)}")

    def toggle_debug_mode(self):
        """Toggle debug mode for extraction."""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            messagebox.showinfo("Debug Mode", "Debug mode enabled. Extraction details will be printed to console.")
        else:
            messagebox.showinfo("Debug Mode", "Debug mode disabled.")

    def export_cooccurrence(self):
        """Export co-occurrence data to CSV."""
        if not hasattr(self, 'bigram_cooccurrence') or not self.bigram_cooccurrence:
            messagebox.showwarning("Export", "No co-occurrence data available.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save co-occurrence data as CSV"
        )
        
        if not file_path:
            return

        try:
            # Choose which co-occurrence data to export
            if self.analysis_type.get() == "bigram":
                cooccur_data = self.bigram_cooccurrence
                analysis_name = "Adjacent Bigram"
            else:
                cooccur_data = self.sentence_cooccurrence
                analysis_name = "Sentence Co-occurrence"

            # Calculate PMI scores
            total_terms = sum(item["freq"] for item in self.term_data)
            term_freq_map = {item["term"].lower(): item["freq"] for item in self.term_data}

            export_data = []
            for (term1, term2), count in cooccur_data.items():
                pmi_score = 0
                if term1 in term_freq_map and term2 in term_freq_map:
                    pmi_score = calculate_pmi(count, term_freq_map[term1], 
                                            term_freq_map[term2], total_terms)
                
                export_data.append({
                    "Term 1": term1.title(),
                    "Term 2": term2.title(),
                    "Co-occurrence Count": count,
                    "PMI Score": round(pmi_score, 4),
                    "Analysis Type": analysis_name,
                    "Term 1 Frequency": term_freq_map.get(term1, 0),
                    "Term 2 Frequency": term_freq_map.get(term2, 0)
                })

            # Sort by co-occurrence count
            export_data.sort(key=lambda x: x["Co-occurrence Count"], reverse=True)

            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            messagebox.showinfo("Export Successful", 
                              f"Exported {len(export_data)} co-occurrence pairs to:\n{file_path}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file:\n{str(e)}")

# -------------------- Application Startup --------------------

def show_loading_dialog(root, message="Loading spaCy model, please wait..."):
    """Show loading dialog while initializing."""
    dialog = tk.Toplevel(root)
    dialog.title("Initializing Term Base Creator")
    dialog.geometry("400x150")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()
    
    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{x}+{y}")
    
    # Add icon/logo area (you can add an icon here if you have one)
    header_frame = tk.Frame(dialog, bg="lightblue", height=40)
    header_frame.pack(fill=tk.X)
    header_frame.pack_propagate(False)
    
    title_label = tk.Label(header_frame, text="Term Base Creator", 
                          font=("Arial", 12, "bold"), bg="lightblue", fg="darkblue")
    title_label.pack(expand=True)
    
    # Loading message
    tk.Label(dialog, text=message, pady=15, font=("Arial", 10)).pack()
    
    # Progress bar
    progress = ttk.Progressbar(dialog, mode='indeterminate', length=300)
    progress.pack(padx=20, pady=10)
    progress.start(10)
    
    # Status label
    status_label = tk.Label(dialog, text="Initializing...", fg="gray", font=("Arial", 8))
    status_label.pack()
    
    root.update()
    return dialog, progress, status_label

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import spacy
    except ImportError:
        missing_deps.append("spacy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import docx
    except ImportError:
        missing_deps.append("python-docx")
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        missing_deps.append("beautifulsoup4")
    
    return missing_deps

def show_dependency_error(missing_deps):
    """Show error dialog for missing dependencies."""
    root = tk.Tk()
    root.withdraw()
    
    deps_text = "\n".join([f"  • {dep}" for dep in missing_deps])
    message = f"""Missing Required Dependencies:

{deps_text}

Please install the missing packages using:
pip install {' '.join(missing_deps)}

Or install all dependencies with:
pip install spacy pandas python-docx beautifulsoup4

After installation, you'll also need to download the spaCy language model:
python -m spacy download en_core_web_md"""
    
    messagebox.showerror("Missing Dependencies", message)
    root.quit()

def main():
    """Main application entry point."""
    # Check dependencies first
    missing_deps = check_dependencies()
    if missing_deps:
        show_dependency_error(missing_deps)
        return
    
    root = tk.Tk()
    root.withdraw()  # Hide main window initially
    
    # Show loading dialog
    loading_dialog, progress_bar, status_label = show_loading_dialog(root)
    
    def update_status(text):
        """Update status in loading dialog."""
        status_label.config(text=text)
        root.update()
    
    def load_model_and_start():
        """Load spaCy model in background thread."""
        try:
            update_status("Loading spaCy model...")
            import spacy
            from spacy.matcher import Matcher
            from spacy.symbols import ORTH
            from spacy.lang.en.stop_words import STOP_WORDS
            
            # Try to load the model
            try:
                nlp = spacy.load("en_core_web_md")
                update_status("Configuring tokenizer...")
                add_tokenizer_exceptions(nlp)
                
            except OSError:
                # If medium model not available, try small model
                update_status("Medium model not found, trying small model...")
                try:
                    nlp = spacy.load("en_core_web_sm")
                    add_tokenizer_exceptions(nlp)
                    root.after(0, lambda: show_model_warning())
                except OSError:
                    root.after(0, lambda: show_model_error())
                    return
            
            update_status("Starting application...")
            root.after(0, lambda: finish_startup(nlp))
            
        except Exception as e:
            root.after(0, lambda: show_general_error(str(e)))
    
    def show_model_warning():
        """Show warning if using small model instead of medium."""
        messagebox.showwarning("Model Notice", 
                             "Using spaCy 'en_core_web_sm' model instead of 'en_core_web_md'.\n\n"
                             "For better term extraction quality, consider installing the medium model:\n"
                             "python -m spacy download en_core_web_md")
    
    def show_model_error():
        """Show error if spaCy model not found."""
        progress_bar.stop()
        loading_dialog.destroy()
        messagebox.showerror("Model Error", 
                           "No spaCy English model found.\n\n"
                           "Please install a spaCy English model with one of:\n"
                           "python -m spacy download en_core_web_md  (recommended)\n"
                           "python -m spacy download en_core_web_sm  (smaller, faster)")
        root.quit()
    
    def show_general_error(error_msg):
        """Show general initialization error."""
        progress_bar.stop()
        loading_dialog.destroy()
        messagebox.showerror("Initialization Error", 
                           f"Failed to initialize application:\n\n{error_msg}")
        root.quit()
    
    def finish_startup(nlp_model):
        """Complete application startup."""
        progress_bar.stop()
        loading_dialog.destroy()
        
        # Configure main window
        root.deiconify()  # Show main window
        root.state('zoomed') if root.tk.call('tk', 'windowingsystem') == 'win32' else root.attributes('-zoomed', True)
        
        # Start the application
        try:
            app = TermExtractorApp(root, nlp_model)
            
            # Show welcome message
            welcome_msg = """Welcome to Term Base Creator!

Getting Started:
1. Click 'Load Corpus' to load your text files
2. Use 'Auto-Extract' to automatically find terms
3. Use 'Manual Search' to find specific terms
4. Validate terms in the 'Term Validation' tab
5. Export your termbase when ready

Tips:
• Load a corpus first before extracting terms
• Use the import feature to work with existing termbases
• Check the co-occurrence tab for term relationships"""
            
            messagebox.showinfo("Welcome", welcome_msg)
            
        except Exception as e:
            messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")
            root.quit()
            return
        
        root.mainloop()
    
    # Start model loading in background
    threading.Thread(target=load_model_and_start, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    main()