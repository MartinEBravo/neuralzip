package main

/*
#include <stddef.h>
#include <stdint.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"math/bits"
	"runtime/cgo"
	"sort"
	"unsafe"

	"github.com/klauspost/compress/fse"
	"github.com/klauspost/compress/huff0"
)

const (
	statusOK         = C.int(0)
	statusError      = C.int(-1)
	codebookCapacity = 256
	maxErrorCapacity = 4096
)

type buildNode struct {
	count  uint32
	parent uint16
	symbol uint8
	bits   uint8
}

// buildCodeTableLimited covers the table-log bounds that klauspost/huff0's
// public builder does not expose: 1-4 and 12 bits.
// It follows Huff0/Zstd's tree and height correction, but intentionally uses
// a stable Go sort: equal-frequency symbols may get different lengths while
// retaining the same weighted cost. klauspost/fse emits the table header.
func buildCodeTableLimited(counts *[codebookCapacity]uint32, target uint8) ([]byte, error) {
	const (
		startNode    = codebookCapacity
		nodeCapacity = 2 * codebookCapacity
		countBarrier = uint32(1 << 30)
	)
	if target < 1 || target > 12 {
		return nil, fmt.Errorf("Huff0 table log must be in [1, 12], got %d", target)
	}

	symbolLen := 0
	nonZero := 0
	var total uint64
	for symbol, count := range counts {
		total += uint64(count)
		if count != 0 {
			symbolLen = symbol + 1
			nonZero++
		}
	}
	if nonZero < 2 {
		return nil, errors.New("Huff0 requires at least two symbols")
	}
	if nonZero > 1<<target {
		return nil, fmt.Errorf("Huff0 table log %d cannot represent %d symbols", target, nonZero)
	}
	if total >= uint64(countBarrier) {
		return nil, errors.New("Huff0 histogram total must be below 2^30")
	}

	var nodes [nodeCapacity]buildNode
	for symbol, count := range counts[:symbolLen] {
		nodes[symbol] = buildNode{count: count, symbol: uint8(symbol)}
	}
	sort.SliceStable(nodes[:symbolLen], func(left, right int) bool {
		return nodes[left].count > nodes[right].count
	})

	lastNonNull := symbolLen - 1
	for nodes[lastNonNull].count == 0 {
		lastNonNull--
	}
	lowSymbol := lastNonNull
	nodeNumber := startNode
	lowNode := nodeNumber
	root := nodeNumber + lowSymbol - 1
	nodes[nodeNumber].count = nodes[lowSymbol].count + nodes[lowSymbol-1].count
	nodes[lowSymbol].parent = uint16(nodeNumber)
	nodes[lowSymbol-1].parent = uint16(nodeNumber)
	nodeNumber++
	lowSymbol -= 2
	for node := nodeNumber; node <= root; node++ {
		nodes[node].count = countBarrier
	}

	takeSmallest := func() int {
		if lowSymbol >= 0 && nodes[lowSymbol].count < nodes[lowNode].count {
			selected := lowSymbol
			lowSymbol--
			return selected
		}
		selected := lowNode
		lowNode++
		return selected
	}
	for nodeNumber <= root {
		first := takeSmallest()
		second := takeSmallest()
		nodes[nodeNumber].count = nodes[first].count + nodes[second].count
		nodes[first].parent = uint16(nodeNumber)
		nodes[second].parent = uint16(nodeNumber)
		nodeNumber++
	}

	nodes[root].bits = 0
	for node := root - 1; node >= startNode; node-- {
		nodes[node].bits = nodes[nodes[node].parent].bits + 1
	}
	for node := 0; node <= lastNonNull; node++ {
		nodes[node].bits = nodes[nodes[node].parent].bits + 1
	}
	tableLog, err := limitCodeLengths(nodes[:], lastNonNull, target)
	if err != nil {
		return nil, err
	}

	var lengths [codebookCapacity]uint8
	for node := 0; node <= lastNonNull; node++ {
		lengths[nodes[node].symbol] = nodes[node].bits
	}
	if err := validateCodeLengths(counts, &lengths, tableLog); err != nil {
		return nil, err
	}
	header, err := serializeCodeLengths(&lengths, symbolLen, tableLog)
	if err != nil {
		return nil, err
	}
	parsedLengths, _, parsedTableLog, err := parseCodeTable(header)
	if err != nil {
		return nil, fmt.Errorf("validate generated Huff0 table: %w", err)
	}
	if parsedTableLog != tableLog || parsedLengths != lengths {
		return nil, errors.New("generated Huff0 table does not round-trip")
	}
	return header, nil
}

func limitCodeLengths(nodes []buildNode, lastNonNull int, target uint8) (uint8, error) {
	largest := nodes[lastNonNull].bits
	if largest <= target {
		return largest, nil
	}

	shift := largest - target
	if shift >= 63 {
		return 0, errors.New("Huff0 code length correction exceeds integer range")
	}
	baseCost := int64(1) << shift
	totalCost := int64(0)
	node := lastNonNull
	for node >= 0 && nodes[node].bits > target {
		totalCost += baseCost - (int64(1) << (largest - nodes[node].bits))
		nodes[node].bits = target
		node--
	}
	for node >= 0 && nodes[node].bits == target {
		node--
	}
	if node < 0 || totalCost <= 0 || totalCost&(baseCost-1) != 0 {
		return 0, errors.New("invalid Huff0 height correction state")
	}
	totalCost >>= shift

	const noSymbol = -1
	rankLast := make([]int, int(target)+2)
	for rank := range rankLast {
		rankLast[rank] = noSymbol
	}
	currentBits := target
	for position := node; position >= 0; position-- {
		if nodes[position].bits >= currentBits {
			continue
		}
		currentBits = nodes[position].bits
		rankLast[int(target-currentBits)] = position
	}

	for totalCost > 0 {
		bitsToDecrease := bits.Len64(uint64(totalCost))
		if bitsToDecrease >= len(rankLast) {
			return 0, errors.New("Huff0 code length correction exceeds rank table")
		}
		for ; bitsToDecrease > 1; bitsToDecrease-- {
			highPosition := rankLast[bitsToDecrease]
			lowPosition := rankLast[bitsToDecrease-1]
			if highPosition == noSymbol {
				continue
			}
			if lowPosition == noSymbol || uint64(nodes[highPosition].count) <= 2*uint64(nodes[lowPosition].count) {
				break
			}
		}
		for bitsToDecrease <= int(target) && rankLast[bitsToDecrease] == noSymbol {
			bitsToDecrease++
		}
		if bitsToDecrease > int(target) || rankLast[bitsToDecrease] == noSymbol {
			return 0, errors.New("cannot enforce Huff0 maximum code length")
		}

		totalCost -= int64(1) << (bitsToDecrease - 1)
		position := rankLast[bitsToDecrease]
		nodes[position].bits++
		if rankLast[bitsToDecrease-1] == noSymbol {
			rankLast[bitsToDecrease-1] = position
		}
		if position == 0 {
			rankLast[bitsToDecrease] = noSymbol
		} else {
			position--
			if nodes[position].bits == target-uint8(bitsToDecrease) {
				rankLast[bitsToDecrease] = position
			} else {
				rankLast[bitsToDecrease] = noSymbol
			}
		}
	}

	for totalCost < 0 {
		if rankLast[1] == noSymbol {
			for node >= 0 && nodes[node].bits == target {
				node--
			}
			if node+1 > lastNonNull {
				return 0, errors.New("cannot correct Huff0 code length overshoot")
			}
			nodes[node+1].bits--
			rankLast[1] = node + 1
			totalCost++
			continue
		}
		position := rankLast[1] + 1
		if position > lastNonNull {
			return 0, errors.New("cannot correct Huff0 code length rank")
		}
		nodes[position].bits--
		rankLast[1] = position
		totalCost++
	}
	return target, nil
}

func validateCodeLengths(
	counts *[codebookCapacity]uint32,
	lengths *[codebookCapacity]uint8,
	tableLog uint8,
) error {
	if tableLog == 0 || tableLog > 12 {
		return fmt.Errorf("invalid generated Huff0 table log %d", tableLog)
	}
	var kraft uint32
	for symbol, length := range lengths {
		if counts[symbol] != 0 && length == 0 {
			return fmt.Errorf("generated Huff0 table omits symbol %d", symbol)
		}
		if counts[symbol] == 0 && length != 0 {
			return fmt.Errorf("generated Huff0 table includes absent symbol %d", symbol)
		}
		if length > tableLog {
			return fmt.Errorf("generated Huff0 code length %d exceeds table log %d", length, tableLog)
		}
		if length != 0 {
			kraft += uint32(1) << (tableLog - length)
		}
	}
	if kraft != uint32(1)<<tableLog {
		return fmt.Errorf("generated Huff0 table violates Kraft equality: %d != %d", kraft, uint32(1)<<tableLog)
	}
	return nil
}

func serializeCodeLengths(
	lengths *[codebookCapacity]uint8,
	symbolLen int,
	tableLog uint8,
) ([]byte, error) {
	maxSymbol := symbolLen - 1
	weights := make([]byte, maxSymbol)
	for symbol := range weights {
		length := lengths[symbol]
		if length != 0 {
			weights[symbol] = tableLog + 1 - length
		}
	}
	if maxSymbol >= 2 {
		compressed, err := fse.Compress(weights, &fse.Scratch{TableLog: 6})
		if err == nil && len(compressed) < symbolLen/2 {
			header := make([]byte, len(compressed)+1)
			header[0] = byte(len(compressed))
			copy(header[1:], compressed)
			return header, nil
		}
	}
	if maxSymbol > 128 {
		return nil, errors.New("Huff0 weights cannot be serialized")
	}
	header := make([]byte, 1+(maxSymbol+1)/2)
	header[0] = 128 | byte(maxSymbol-1)
	for symbol := 0; symbol < maxSymbol; symbol += 2 {
		header[1+symbol/2] = weights[symbol] << 4
		if symbol+1 < maxSymbol {
			header[1+symbol/2] |= weights[symbol+1]
		}
	}
	return header, nil
}

type tableHandle struct {
	decoder   *huff0.Decoder
	canonical *canonicalTable
}

type canonicalTable struct {
	lengths [codebookCapacity]uint8
	values  [codebookCapacity]uint16
	maxBits uint8
	decode  map[uint32]byte
}

func newCanonicalTable(
	lengths [codebookCapacity]uint8,
	values [codebookCapacity]uint16,
	maxBits uint8,
) *canonicalTable {
	table := &canonicalTable{
		lengths: lengths,
		values:  values,
		maxBits: maxBits,
		decode:  make(map[uint32]byte, codebookCapacity),
	}
	for symbol, length := range lengths {
		if length != 0 {
			table.decode[uint32(length)<<16|uint32(values[symbol])] = byte(symbol)
		}
	}
	return table
}

func newTableHandle(header []byte) (*tableHandle, error) {
	lengths, values, maxBits, err := parseCodeTable(header)
	if err != nil {
		return nil, err
	}
	if maxBits > 11 {
		return &tableHandle{
			canonical: newCanonicalTable(lengths, values, maxBits),
		}, nil
	}
	decoderScratch, remain, err := huff0.ReadTable(header, nil)
	if err != nil {
		return nil, fmt.Errorf("read generated Huff0 table: %w", err)
	}
	if len(remain) != 0 {
		return nil, errors.New("Huff0 table contains trailing bytes")
	}
	return &tableHandle{decoder: decoderScratch.Decoder()}, nil
}

func tableValue(value C.uintptr_t) (table *tableHandle, err error) {
	if value == 0 {
		return nil, errors.New("invalid zero Huff0 table handle")
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			table = nil
			err = fmt.Errorf("invalid Huff0 table handle: %v", recovered)
		}
	}()
	table, ok := cgo.Handle(value).Value().(*tableHandle)
	if !ok || table == nil {
		return nil, errors.New("invalid Huff0 table handle type")
	}
	return table, nil
}

func inputBytes(pointer *C.uint8_t, size C.size_t) ([]byte, error) {
	if size == 0 {
		return nil, nil
	}
	if pointer == nil {
		return nil, errors.New("non-empty input has a null pointer")
	}
	if uint64(size) > uint64(^uint(0)>>1) {
		return nil, errors.New("input size exceeds the Go address space")
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(pointer)), int(size)), nil
}

func outputBytes(pointer *C.uint8_t, size C.size_t) ([]byte, error) {
	if size == 0 {
		return nil, nil
	}
	if pointer == nil {
		return nil, errors.New("non-empty output has a null pointer")
	}
	if uint64(size) > uint64(^uint(0)>>1) {
		return nil, errors.New("output size exceeds the Go address space")
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(pointer)), int(size)), nil
}

func writeError(destination *C.char, capacity C.size_t, err error) {
	if destination == nil || capacity == 0 {
		return
	}
	length := min(uint64(capacity), uint64(maxErrorCapacity))
	buffer := unsafe.Slice((*byte)(unsafe.Pointer(destination)), int(length))
	for index := range buffer {
		buffer[index] = 0
	}
	if err == nil || len(buffer) == 1 {
		return
	}
	message := []byte(err.Error())
	if len(message) >= len(buffer) {
		message = message[:len(buffer)-1]
	}
	copy(buffer, message)
}

func guarded(errorDestination *C.char, errorCapacity C.size_t, operation func() error) (status C.int) {
	status = statusError
	defer func() {
		if recovered := recover(); recovered != nil {
			writeError(errorDestination, errorCapacity, fmt.Errorf("Huff0 panic: %v", recovered))
		}
	}()
	if err := operation(); err != nil {
		writeError(errorDestination, errorCapacity, err)
		return statusError
	}
	writeError(errorDestination, errorCapacity, nil)
	return statusOK
}

func copyResult(destination *C.uint8_t, capacity C.size_t, result []byte, outputSize *C.size_t) error {
	if outputSize == nil {
		return errors.New("output size pointer is null")
	}
	*outputSize = C.size_t(len(result))
	if C.size_t(len(result)) > capacity {
		return fmt.Errorf("Huff0 output buffer is too small: need %d bytes", len(result))
	}
	buffer, err := outputBytes(destination, capacity)
	if err != nil {
		return err
	}
	copy(buffer, result)
	return nil
}

//export nzip_huff0_build_from_counts
func nzip_huff0_build_from_counts(
	countsPointer *C.uint32_t,
	maxBits C.uint8_t,
	headerOutput *C.uint8_t,
	headerCapacity C.size_t,
	headerSize *C.size_t,
	errorDestination *C.char,
	errorCapacity C.size_t,
) C.int {
	return guarded(errorDestination, errorCapacity, func() error {
		if countsPointer == nil || headerSize == nil {
			return errors.New("Huff0 table output or histogram pointer is null")
		}
		if maxBits < 1 || maxBits > 12 {
			return fmt.Errorf("Huff0 table log must be in [1, 12], got %d", maxBits)
		}
		input := unsafe.Slice((*C.uint32_t)(unsafe.Pointer(countsPointer)), codebookCapacity)
		var counts [codebookCapacity]uint32
		for index, count := range input {
			counts[index] = uint32(count)
		}

		var header []byte
		var err error
		if maxBits < 5 || maxBits == 12 {
			header, err = buildCodeTableLimited(&counts, uint8(maxBits))
		} else {
			seed := &huff0.Scratch{TableLog: uint8(maxBits)}
			if err = seed.BuildCTable(&counts); err == nil {
				header, err = seed.AppendTable(nil)
			}
		}
		if err != nil {
			return fmt.Errorf("build Huff0 table: %w", err)
		}
		_, _, actualMaxBits, err := parseCodeTable(header)
		if err != nil {
			return err
		}
		if actualMaxBits > uint8(maxBits) {
			return fmt.Errorf(
				"Huff0 table requires %d bits, exceeding requested limit %d",
				actualMaxBits,
				maxBits,
			)
		}
		if err := copyResult(headerOutput, headerCapacity, header, headerSize); err != nil {
			return err
		}
		return nil
	})
}

//export nzip_huff0_build_from_header
func nzip_huff0_build_from_header(
	headerPointer *C.uint8_t,
	headerLength C.size_t,
	handleOutput *C.uintptr_t,
	errorDestination *C.char,
	errorCapacity C.size_t,
) C.int {
	return guarded(errorDestination, errorCapacity, func() error {
		if handleOutput == nil {
			return errors.New("Huff0 table handle output is null")
		}
		headerView, err := inputBytes(headerPointer, headerLength)
		if err != nil {
			return err
		}
		header := append([]byte(nil), headerView...)
		table, err := newTableHandle(header)
		if err != nil {
			return err
		}
		*handleOutput = C.uintptr_t(cgo.NewHandle(table))
		return nil
	})
}

//export nzip_huff0_release
func nzip_huff0_release(value C.uintptr_t) {
	defer func() { _ = recover() }()
	if value != 0 {
		cgo.Handle(value).Delete()
	}
}

type reverseBitReader struct {
	source   []byte
	position int
}

func newReverseBitReader(source []byte) (*reverseBitReader, error) {
	if len(source) == 0 || source[len(source)-1] == 0 {
		return nil, errors.New("corrupt Huff0 stream end marker")
	}
	endMarker := bits.Len8(source[len(source)-1]) - 1
	return &reverseBitReader{
		source:   source,
		position: (len(source)-1)*8 + endMarker - 1,
	}, nil
}

func (reader *reverseBitReader) read() (uint16, bool) {
	if reader.position < 0 {
		return 0, false
	}
	position := reader.position
	reader.position--
	return uint16((reader.source[position/8] >> (position & 7)) & 1), true
}

func decodeCanonical1X(
	source []byte,
	destination []byte,
	table *canonicalTable,
) error {
	reader, err := newReverseBitReader(source)
	if err != nil {
		return err
	}
	for index := range destination {
		var value uint16
		matched := false
		for length := uint8(1); length <= table.maxBits; length++ {
			bit, ok := reader.read()
			if !ok {
				return errors.New("truncated Huff0 stream")
			}
			value = value<<1 | bit
			if symbol, ok := table.decode[uint32(length)<<16|uint32(value)]; ok {
				destination[index] = symbol
				matched = true
				break
			}
		}
		if !matched {
			return errors.New("invalid Huff0 symbol code")
		}
	}
	if reader.position != -1 {
		return errors.New("Huff0 stream contains unused bits")
	}
	return nil
}

func decodeCanonical4X(
	source []byte,
	destination []byte,
	table *canonicalTable,
) error {
	if len(source) < 6 {
		return errors.New("truncated Huff0 four-stream header")
	}
	sizes := [4]int{
		int(source[0]) | int(source[1])<<8,
		int(source[2]) | int(source[3])<<8,
		int(source[4]) | int(source[5])<<8,
	}
	sizes[3] = len(source) - 6 - sizes[0] - sizes[1] - sizes[2]
	if sizes[0] <= 0 || sizes[1] <= 0 || sizes[2] <= 0 || sizes[3] <= 0 {
		return errors.New("invalid Huff0 four-stream lengths")
	}
	segmentSize := (len(destination) + 3) / 4
	sourceOffset := 6
	destinationOffset := 0
	for stream, size := range sizes {
		sourceEnd := sourceOffset + size
		if sourceEnd > len(source) {
			return errors.New("truncated Huff0 four-stream payload")
		}
		destinationEnd := min(len(destination), destinationOffset+segmentSize)
		if err := decodeCanonical1X(
			source[sourceOffset:sourceEnd],
			destination[destinationOffset:destinationEnd],
			table,
		); err != nil {
			return fmt.Errorf("decode Huff0 stream %d: %w", stream, err)
		}
		sourceOffset = sourceEnd
		destinationOffset = destinationEnd
	}
	if sourceOffset != len(source) || destinationOffset != len(destination) {
		return errors.New("Huff0 four-stream size mismatch")
	}
	return nil
}

func decodeWithTable(table *tableHandle, source []byte, destination []byte, fourStreams bool) error {
	if table.canonical != nil {
		if fourStreams {
			return decodeCanonical4X(source, destination, table.canonical)
		}
		return decodeCanonical1X(source, destination, table.canonical)
	}
	output := destination[:0:len(destination)]
	var err error
	if fourStreams {
		output, err = table.decoder.Decompress4X(output, source)
	} else {
		output, err = table.decoder.Decompress1X(output, source)
	}
	if err != nil {
		return err
	}
	if len(output) != len(destination) {
		return fmt.Errorf("Huff0 decoded size mismatch: %d != %d", len(output), len(destination))
	}
	return nil
}

//export nzip_huff0_decode
func nzip_huff0_decode(
	value C.uintptr_t,
	sourcePointer *C.uint8_t,
	sourceLength C.size_t,
	destination *C.uint8_t,
	destinationLength C.size_t,
	fourStreams C.int,
	errorDestination *C.char,
	errorCapacity C.size_t,
) C.int {
	return guarded(errorDestination, errorCapacity, func() error {
		table, err := tableValue(value)
		if err != nil {
			return err
		}
		source, err := inputBytes(sourcePointer, sourceLength)
		if err != nil {
			return err
		}
		output, err := outputBytes(destination, destinationLength)
		if err != nil {
			return err
		}
		if err := decodeWithTable(table, source, output, fourStreams != 0); err != nil {
			return fmt.Errorf("decode Huff0 block: %w", err)
		}
		return nil
	})
}

//export nzip_huff0_decompress_local
func nzip_huff0_decompress_local(
	sourcePointer *C.uint8_t,
	sourceLength C.size_t,
	destination *C.uint8_t,
	destinationLength C.size_t,
	errorDestination *C.char,
	errorCapacity C.size_t,
) C.int {
	return guarded(errorDestination, errorCapacity, func() error {
		source, err := inputBytes(sourcePointer, sourceLength)
		if err != nil {
			return err
		}
		output, err := outputBytes(destination, destinationLength)
		if err != nil {
			return err
		}
		if len(source) < 2 {
			return errors.New("local Huff0 block is too small")
		}
		encodedSize := int(source[0])
		headerLength := 1 + encodedSize
		if encodedSize >= 128 {
			weightCount := encodedSize - 127
			headerLength = 1 + (weightCount+1)/2
		}
		if headerLength >= len(source) {
			return errors.New("local Huff0 block has an invalid table size")
		}
		table, err := newTableHandle(source[:headerLength])
		if err != nil {
			return fmt.Errorf("read local Huff0 table: %w", err)
		}
		if err := decodeWithTable(table, source[headerLength:], output, true); err != nil {
			return fmt.Errorf("decode local Huff0 block: %w", err)
		}
		return nil
	})
}

func parseCodeTable(header []byte) (lengths [codebookCapacity]uint8, values [codebookCapacity]uint16, tableLog uint8, err error) {
	if len(header) <= 1 {
		return lengths, values, 0, errors.New("Huff0 table header is too small")
	}
	encodedSize := int(header[0])
	payload := header[1:]
	weights := make([]byte, 0, codebookCapacity)
	if encodedSize >= 128 {
		weightCount := encodedSize - 127
		byteCount := (weightCount + 1) / 2
		if byteCount > len(payload) || 1+byteCount != len(header) {
			return lengths, values, 0, errors.New("invalid raw Huff0 table header size")
		}
		for index := 0; index < weightCount; index++ {
			packed := payload[index/2]
			if index&1 == 0 {
				weights = append(weights, packed>>4)
			} else {
				weights = append(weights, packed&15)
			}
		}
	} else {
		if encodedSize > len(payload) || 1+encodedSize != len(header) {
			return lengths, values, 0, errors.New("invalid FSE Huff0 table header size")
		}
		scratch := &fse.Scratch{DecompressLimit: 255}
		decoded, decodeErr := fse.Decompress(payload[:encodedSize], scratch)
		if decodeErr != nil {
			return lengths, values, 0, fmt.Errorf("decode Huff0 table weights: %w", decodeErr)
		}
		weights = append(weights, decoded...)
	}
	if len(weights) == 0 || len(weights) > 255 {
		return lengths, values, 0, errors.New("invalid Huff0 table weight count")
	}

	var rankStats [16]uint32
	var weightTotal uint32
	for _, weight := range weights {
		if weight > 12 {
			return lengths, values, 0, errors.New("Huff0 table weight exceeds 12 bits")
		}
		rankStats[weight]++
		weightTotal += (uint32(1) << weight) >> 1
	}
	if weightTotal == 0 {
		return lengths, values, 0, errors.New("Huff0 table weights are empty")
	}
	tableLog = uint8(bits.Len32(weightTotal))
	if tableLog > 12 {
		return lengths, values, 0, errors.New("Huff0 table log exceeds 12 bits")
	}
	total := uint32(1) << tableLog
	if weightTotal >= total {
		return lengths, values, 0, errors.New("invalid implied Huff0 table weight")
	}
	remainder := total - weightTotal
	if remainder&(remainder-1) != 0 {
		return lengths, values, 0, errors.New("implied Huff0 table weight is not a power of two")
	}
	lastWeight := uint8(bits.Len32(remainder))
	weights = append(weights, lastWeight)
	rankStats[lastWeight]++
	if rankStats[1] < 2 || rankStats[1]&1 != 0 {
		return lengths, values, 0, errors.New("invalid minimum Huff0 table rank")
	}

	var nextRankStart uint32
	for rank := uint8(1); rank <= tableLog; rank++ {
		current := nextRankStart
		nextRankStart += rankStats[rank] << (rank - 1)
		rankStats[rank] = current
	}
	for symbol, weight := range weights {
		if weight == 0 {
			continue
		}
		lengths[symbol] = tableLog + 1 - weight
		values[symbol] = uint16(rankStats[weight] >> (weight - 1))
		rankStats[weight] += uint32(1) << (weight - 1)
	}
	return lengths, values, tableLog, nil
}

//export nzip_huff0_export_table
func nzip_huff0_export_table(
	headerPointer *C.uint8_t,
	headerLength C.size_t,
	lengthOutput *C.uint8_t,
	valueOutput *C.uint16_t,
	maxBitsOutput *C.uint8_t,
	errorDestination *C.char,
	errorCapacity C.size_t,
) C.int {
	return guarded(errorDestination, errorCapacity, func() error {
		if lengthOutput == nil || valueOutput == nil || maxBitsOutput == nil {
			return errors.New("Huff0 code table output pointer is null")
		}
		header, err := inputBytes(headerPointer, headerLength)
		if err != nil {
			return err
		}
		lengths, values, maxBits, err := parseCodeTable(header)
		if err != nil {
			return err
		}
		lengthDestination := unsafe.Slice((*byte)(unsafe.Pointer(lengthOutput)), codebookCapacity)
		valueDestination := unsafe.Slice((*C.uint16_t)(unsafe.Pointer(valueOutput)), codebookCapacity)
		copy(lengthDestination, lengths[:])
		for index, value := range values {
			valueDestination[index] = C.uint16_t(value)
		}
		*maxBitsOutput = C.uint8_t(maxBits)
		return nil
	})
}

func main() {}
