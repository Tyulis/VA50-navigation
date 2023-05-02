#ifndef _TRAJECTORY_COMMON_H
#define _TRAJECTORY_COMMON_H

#include <string>

class Direction {
	public:
		enum _Value : int {
			None = 0,
			Forward = 1,
			Left = 2,
			ForwardLeft = 3,
			Right = 4,
			ForwardRight = 5,
			LeftRight = 6,
			All = 7,
		};

		inline constexpr Direction() : m_value(None) {}
		inline constexpr Direction(_Value value) : m_value(value) {}
		inline constexpr Direction(int value) : m_value(static_cast<_Value>(value)) {}

		inline constexpr bool operator==(Direction b) { return m_value == b.m_value; }
		inline constexpr bool operator!=(Direction b) { return m_value != b.m_value; }
		inline constexpr bool operator==(_Value b) { return m_value == b; }
		inline constexpr bool operator!=(_Value b) { return m_value != b; }

		inline constexpr Direction operator~ () { return (_Value)~(int)m_value; }
		inline constexpr Direction operator| (Direction b) { return (_Value)((int)m_value | (int)b.m_value); }
		inline constexpr Direction operator& (Direction b) { return (_Value)((int)m_value & (int)b.m_value); }
		inline constexpr Direction operator^ (Direction b) { return (_Value)((int)m_value ^ (int)b.m_value); }
		inline constexpr Direction& operator|= (Direction b) { m_value = (_Value)((int)m_value | (int)b.m_value); return *this; }
		inline constexpr Direction& operator&= (Direction b) { m_value = (_Value)((int&)m_value & (int)b.m_value); return *this; }
		inline constexpr Direction& operator^= (Direction b) { m_value = (_Value)((int&)m_value ^ (int)b.m_value); return *this; }
		inline constexpr Direction operator| (_Value b) { return (_Value)((int)m_value | (int)b); }
		inline constexpr Direction operator& (_Value b) { return (_Value)((int)m_value & (int)b); }
		inline constexpr Direction operator^ (_Value b) { return (_Value)((int)m_value ^ (int)b); }
		inline constexpr Direction& operator|= (_Value b) { m_value = (_Value)((int)m_value | (int)b); return *this; }
		inline constexpr Direction& operator&= (_Value b) { m_value = (_Value)((int&)m_value & (int)b); return *this; }
		inline constexpr Direction& operator^= (_Value b) { m_value = (_Value)((int&)m_value ^ (int)b); return *this; }
		inline constexpr explicit operator bool() { return m_value != None; }
		inline constexpr operator _Value() { return m_value; }
	
	private:
		_Value m_value;

};

/* Bitflags operations */

namespace std {
	inline string to_string(Direction value) {
		switch (value) {
			case Direction::None        : return "None";
			case Direction::Forward     : return "Forward";
			case Direction::Left        : return "Left";
			case Direction::Right       : return "Right";
			case Direction::ForwardLeft : return "Forward | Left";
			case Direction::ForwardRight: return "Forward | Right";
			case Direction::LeftRight   : return "Left | Right";
			case Direction::All         : return "Forward | Left | Right";
		}
	}
}

#endif