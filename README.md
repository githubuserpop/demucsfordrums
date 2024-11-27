# Demucs for Individual Drum Separation

A specialized adaptation of Hybrid Demucs focused on separating individual drum components from drum stems.

![Demucs Logo](demucs.png)

## Overview

This project extends the original Demucs music source separation model to focus specifically on drum stem separation. Instead of the conventional VDBO (Vocals, Drums, Bass, Other) separation, this adaptation specializes in breaking down drum stems into individual components like kicks, snares, hi-hats, and more.

## Features

- Individual drum component separation from drum stems
- Support for multiple drum categories:
  - Kicks
  - Snares
  - Hi-hats
  - Open hats
  - Claps
  - Percs
  - FX
  - Other percussion elements
- Built on the powerful Hybrid Demucs architecture
- Python-based with both CPU and GPU support

## Dataset

The model is trained on a comprehensive dataset consisting of:
- 1,860 custom-produced stems organized by drum type
- 25,418 high-quality drum samples from various producers
- 4,763 track stems divided into:
  - Drum stems
  - Non-drum stems
  - Original source tracks

## Development Roadmap

1. **Enhanced Architecture**
   - Implementation of additional output layers
   - Pattern recognition system integration
   - Multiwrap configuration updates

2. **Training Pipeline**
   - Custom training workflow adaptation
   - Dataset integration
   - Model evaluation and optimization

3. **Quality Assurance**
   - Separation quality metrics
   - Output testing framework
   - Performance benchmarking

## Future Development

- **Direct Track Processing**: Implement direct processing of full tracks without requiring pre-separated drum stems
- **Extended Component Recognition**: Expand the range of recognizable drum components
- **Real-time Processing**: Explore possibilities for real-time separation

## Motivation

This project was born from a need to isolate specific drum sounds from existing tracks. While traditional source separation tools focus on broad category separation (VDBO), this adaptation provides granular control over drum component isolation, enabling producers to:
- Extract specific drum sounds for sampling
- Study and analyze drum patterns
- Remix and repurpose drum elements

## Contributing

We welcome contributions! Whether it's:
- Adding new features
- Improving separation quality
- Expanding the training dataset
- Bug fixes and optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the original Demucs project by Meta AI Research

## Contact

For questions, suggestions, or discussions, please open an issue in the GitHub repository.