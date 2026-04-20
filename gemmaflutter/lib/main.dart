import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'package:flutter_dotenv/flutter_dotenv.dart';


void main() async {
  await dotenv.load(fileName: ".env");
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gemma Robot Controller',
      home: Scaffold(
        appBar: AppBar(title: const Text('Robot Arm Controller')),
        body: const Center(
          child: VoiceCommandButton(),
        ),
      ),
    );
  }
}

class VoiceCommandButton extends StatefulWidget {
  const VoiceCommandButton({super.key});

  @override
  State<VoiceCommandButton> createState() => _VoiceCommandButtonState();
}

class _VoiceCommandButtonState extends State<VoiceCommandButton> {
  final SpeechToText _speech = SpeechToText();
  bool _isListening = false;
  bool _isAvailable = false;
  String _currentWords = '';
  String _response = '';

  @override
  void initState() {
    super.initState();
    _initSpeech();
  }

  Future<void> _initSpeech() async {
    _isAvailable = await _speech.initialize();
    setState(() {});
  }

  Future<void> _startListening() async {
    if (!_isAvailable) return;
    await _speech.listen(
      onResult: _onResult,
      listenFor: const Duration(seconds: 10),
      pauseFor: const Duration(seconds: 2),
      partialResults: true,
      listenMode: ListenMode.dictation,
    );
    setState(() => _isListening = true);
  }

  Future<void> _stopListening() async {
    await _speech.stop();
    setState(() => _isListening = false);
  }

  void _onResult(SpeechRecognitionResult result) {
    setState(() => _currentWords = result.recognizedWords);
    if (result.finalResult && _currentWords.isNotEmpty) {
      _sendToGemma(_currentWords);
      _currentWords = '';
    }
  }
  
  final vmIp = dotenv.env['VM_EXTERNAL_IP'] ?? '';
  final localIp = dotenv.env['LOCAL_IP'] ?? '';

  Future<void> _sendToGemma(String transcript) async {
    try {
      final response = await http.post(
        Uri.parse('http://$vmIp:8080/v1/chat/completions'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'messages': [
            {
              'role': 'system',
              'content': 'You are a robot arm controller. Convert the user\'s voice command into a JSON object with exactly these three fields: task_description (a concise instruction in the style of robot training data, e.g. "pick up the red cup on the left side of the table"), object (the target object), and action (the action to perform). Respond ONLY with valid JSON. No explanation, no markdown, no code fences.'
            },
            {'role': 'user', 'content': transcript}
          ],
          'temperature': 0.2,
          'max_tokens': 200,
        }),
      );
      final data = jsonDecode(response.body);
      String content = data['choices'][0]['message']['content'];
      content = content.replaceAll('```json', '').replaceAll('```', '').trim();
      setState(() => _response = content);

      await http.post(
        Uri.parse('http://$localIp:5000/save'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'instruction': content, 'transcript': transcript}),
      );
    } catch (e) {
      setState(() => _response = 'Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (_currentWords.isNotEmpty)
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text('Listening: $_currentWords',
                style: const TextStyle(fontSize: 16)),
          ),
        if (_response.isNotEmpty)
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text('Instructions: $_response',
                style: const TextStyle(fontSize: 14, color: Colors.green)),
          ),
        GestureDetector(
          onLongPressStart: (_) => _startListening(),
          onLongPressEnd: (_) => _stopListening(),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            width: _isListening ? 80 : 64,
            height: _isListening ? 80 : 64,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _isListening ? Colors.red : Colors.blue,
            ),
            child: Icon(
              _isListening ? Icons.mic : Icons.mic_none,
              color: Colors.white,
              size: 32,
            ),
          ),
        ),
        Text(_isListening ? 'Listening...' : 'Hold to speak'),
      ],
    );
  }
}