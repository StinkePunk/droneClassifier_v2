def split_into_chunks(audio, chunk_length=48000):
    """
    Teilt das Audiosignal in gleichmäßige Chunks der Länge `chunk_length`.
    Verwirft den letzten Teil, falls er kürzer ist als `chunk_length`.
    """
    num_chunks = len(audio) // chunk_length
    chunks = []
    for i in range(num_chunks):
        chunk = audio[i * chunk_length:(i + 1) * chunk_length]
        chunks.append(chunk)
    return chunks

def split_data_and_labels(audio_files, labels, chunk_length=48000):
    """
    Teilt Audiodaten in Chunks auf und passt die Labels entsprechend an.

    Parameters:
    - audio_files: Liste von Audiodaten.
    - labels: Liste der Labels für die Audiodaten.
    - chunk_length: Länge jedes Chunks in Samples.

    Returns:
    - split_audio: Liste der Chunks.
    - split_labels: Liste der Labels für jeden Chunk.
    """
    split_audio = []
    split_labels = []
    
    for audio, label in zip(audio_files, labels):
        chunks = split_into_chunks(audio, chunk_length=chunk_length)
        split_audio.extend(chunks)
        split_labels.extend([label] * len(chunks))  # Dupliziere das Label für jeden Chunk
    
    return split_audio, split_labels