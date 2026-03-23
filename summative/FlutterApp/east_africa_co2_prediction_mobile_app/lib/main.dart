import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const Co2App());

const _bg = Color(0xFF0A1E3C);
const _yellow = Color(0xFFFFC107);
const _surface = Color(0xFF112244);
const _baseUrl = 'https://linear-regression-model-wk1e.onrender.com';

class Co2App extends StatelessWidget {
  const Co2App({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'East Africa CO₂ Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: _bg,
        colorScheme: const ColorScheme.dark(primary: _yellow, surface: _surface),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: _surface,
          labelStyle: const TextStyle(color: Colors.white70, fontSize: 13),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: const BorderSide(color: Colors.white24)),
          enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: const BorderSide(color: Colors.white24)),
          focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(10), borderSide: const BorderSide(color: _yellow, width: 1.5)),
          errorStyle: const TextStyle(fontSize: 11),
          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

// Holds controllers for one prediction row
class _RowControllers {
  final country = TextEditingController();
  final year = TextEditingController();
  final population = TextEditingController();
  final transport = TextEditingController();
  final manufacturing = TextEditingController();
  final electricity = TextEditingController();
  final building = TextEditingController();

  void dispose() {
    for (final c in [country, year, population, transport, manufacturing, electricity, building]) {
      c.dispose();
    }
  }

  Map<String, dynamic> toJson() => {
        'country': country.text.trim(),
        'year': int.parse(year.text.trim()),
        'population': double.parse(population.text.trim()),
        'transportation_mt': double.parse(transport.text.trim()),
        'manufacturing_construction_mt': double.parse(manufacturing.text.trim()),
        'electricity_heat_mt': double.parse(electricity.text.trim()),
        'building_mt': double.parse(building.text.trim()),
      };
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  bool _batchMode = false;
  bool _loading = false;
  String? _singleResult;
  List<double>? _batchResults;
  String? _errorMsg;

  // Single mode uses index 0, batch mode uses all
  final List<_RowControllers> _rows = [_RowControllers()];

  @override
  void dispose() {
    for (final r in _rows) {
      r.dispose();
    }
    super.dispose();
  }

  void _addRow() => setState(() => _rows.add(_RowControllers()));

  void _removeRow(int index) {
    if (_rows.length == 1) return;
    setState(() {
      _rows[index].dispose();
      _rows.removeAt(index);
    });
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() { _loading = true; _singleResult = null; _batchResults = null; _errorMsg = null; });

    try {
      final http.Response response;

      if (!_batchMode) {
        response = await http.post(
          Uri.parse('$_baseUrl/predict'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(_rows[0].toJson()),
        ).timeout(const Duration(seconds: 30));
      } else {
        response = await http.post(
          Uri.parse('$_baseUrl/predict/batch'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'rows': _rows.map((r) => r.toJson()).toList()}),
        ).timeout(const Duration(seconds: 30));
      }

      final data = jsonDecode(response.body);

      if (response.statusCode == 200) {
        if (!_batchMode) {
          final value = (data['prediction_mt'] as num).toStringAsFixed(4);
          setState(() { _singleResult = '$value Mt'; });
        } else {
          final list = (data['predictions_mt'] as List).map((e) => (e as num).toDouble()).toList();
          setState(() { _batchResults = list; });
        }
      } else {
        final detail = data['detail'] ?? 'Unexpected error (${response.statusCode})';
        setState(() { _errorMsg = detail.toString(); });
      }
    } catch (e) {
      setState(() { _errorMsg = 'Request failed: $e'; });
    } finally {
      setState(() { _loading = false; });
    }
  }

  Widget _field({
    required TextEditingController ctrl,
    required String label,
    required String hint,
    bool isText = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: TextFormField(
        controller: ctrl,
        keyboardType: isText ? TextInputType.text : TextInputType.number,
        style: const TextStyle(color: Colors.white, fontSize: 14),
        decoration: InputDecoration(labelText: label, hintText: hint, hintStyle: const TextStyle(color: Colors.white38)),
        validator: (v) {
          if (v == null || v.trim().isEmpty) return '$label is required';
          if (!isText) {
            if (double.tryParse(v.trim()) == null) return 'Enter a valid number';
            if (double.parse(v.trim()) < 0) return 'Value must be ≥ 0';
          }
          return null;
        },
      ),
    );
  }

  Widget _buildRowCard(_RowControllers row, int index) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 4),
      decoration: BoxDecoration(
        color: _surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_batchMode)
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Entry ${index + 1}', style: const TextStyle(color: _yellow, fontWeight: FontWeight.bold, fontSize: 13)),
                if (_rows.length > 1)
                  GestureDetector(
                    onTap: () => _removeRow(index),
                    child: const Icon(Icons.remove_circle_outline, color: Colors.redAccent, size: 20),
                  ),
              ],
            ),
          if (_batchMode) const SizedBox(height: 10),
          _field(ctrl: row.country, label: 'Country', hint: 'e.g. Kenya', isText: true),
          _field(ctrl: row.year, label: 'Year', hint: 'e.g. 2020'),
          _field(ctrl: row.population, label: 'Population', hint: 'e.g. 53771300'),
          _field(ctrl: row.transport, label: 'Transportation (Mt)', hint: 'e.g. 5.1'),
          _field(ctrl: row.manufacturing, label: 'Manufacturing / Construction (Mt)', hint: 'e.g. 2.3'),
          _field(ctrl: row.electricity, label: 'Electricity / Heat (Mt)', hint: 'e.g. 3.8'),
          _field(ctrl: row.building, label: 'Building (Mt)', hint: 'e.g. 1.7'),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: _surface,
        title: const Text('East Africa CO₂ Emission Predictor',
            style: TextStyle(color: _yellow, fontWeight: FontWeight.bold, fontSize: 18)),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Mode toggle
                Container(
                  decoration: BoxDecoration(
                    color: _surface,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: Row(
                    children: [
                      _modeTab('Single Entry', !_batchMode, () => setState(() {
                        _batchMode = false;
                        _singleResult = null; _batchResults = null; _errorMsg = null;
                        while (_rows.length > 1) {
                          _rows.last.dispose();
                          _rows.removeLast();
                        }
                      })),
                      _modeTab('Multiple Entries', _batchMode, () => setState(() {
                        _batchMode = true;
                        _singleResult = null; _batchResults = null; _errorMsg = null;
                      })),
                    ],
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  _batchMode
                      ? 'Add multiple entries and predict all at once'
                      : 'Enter emission indicators to predict total CO₂ emissions (Mt)',
                  style: const TextStyle(color: Colors.white60, fontSize: 12),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),

                // Row cards
                ...List.generate(_rows.length, (i) => _buildRowCard(_rows[i], i)),

                // Add row button (batch only)
                if (_batchMode) ...[
                  OutlinedButton.icon(
                    onPressed: _addRow,
                    icon: const Icon(Icons.add, color: _yellow, size: 18),
                    label: const Text('Add Entry', style: TextStyle(color: _yellow)),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: _yellow),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                  const SizedBox(height: 14),
                ],

                // Predict button
                SizedBox(
                  height: 50,
                  child: ElevatedButton(
                    onPressed: _loading ? null : _predict,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _yellow,
                      foregroundColor: _bg,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    child: _loading
                        ? const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.5, color: _bg))
                        : const Text('Predict'),
                  ),
                ),

                // Error display
                if (_errorMsg != null) ...[
                  const SizedBox(height: 24),
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: _surface,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.redAccent, width: 1.5),
                    ),
                    child: Column(
                      children: [
                        const Text('Error', style: TextStyle(color: Colors.redAccent, fontWeight: FontWeight.bold, fontSize: 13)),
                        const SizedBox(height: 8),
                        Text(_errorMsg!, style: TextStyle(color: Colors.red[300], fontSize: 13), textAlign: TextAlign.center),
                      ],
                    ),
                  ),
                ],

                // Single result display
                if (_singleResult != null) ...[
                  const SizedBox(height: 24),
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: _surface,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _yellow, width: 1.5),
                    ),
                    child: Column(
                      children: [
                        const Text('Predicted CO₂ Emission', style: TextStyle(color: _yellow, fontWeight: FontWeight.bold, fontSize: 13)),
                        const SizedBox(height: 8),
                        Text(_singleResult!, style: const TextStyle(color: Colors.white, fontSize: 26, fontWeight: FontWeight.bold), textAlign: TextAlign.center),
                      ],
                    ),
                  ),
                ],

                // Batch results display
                if (_batchResults != null) ...[
                  const SizedBox(height: 24),
                  Container(
                    decoration: BoxDecoration(
                      color: _surface,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _yellow, width: 1.5),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const Padding(
                          padding: EdgeInsets.fromLTRB(16, 14, 16, 8),
                          child: Text('Predictions', style: TextStyle(color: _yellow, fontWeight: FontWeight.bold, fontSize: 13)),
                        ),
                        const Divider(color: Colors.white12, height: 1),
                        ...List.generate(_batchResults!.length, (i) => Theme(
                          data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                          child: ExpansionTile(
                            initiallyExpanded: true,
                            leading: const Icon(Icons.arrow_forward_ios, color: _yellow, size: 14),
                            title: Text('Entry ${i + 1}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 14)),
                            children: [
                              Padding(
                                padding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    const Text('Predicted CO₂ Emission', style: TextStyle(color: Colors.white60, fontSize: 13)),
                                    Text('${_batchResults![i].toStringAsFixed(4)} Mt', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15)),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        )),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _modeTab(String label, bool active, VoidCallback onTap) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: active ? _yellow : Colors.transparent,
            borderRadius: BorderRadius.circular(9),
          ),
          alignment: Alignment.center,
          child: Text(
            label,
            style: TextStyle(
              color: active ? _bg : Colors.white60,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
        ),
      ),
    );
  }
}
