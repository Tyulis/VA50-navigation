#ifndef _TRAJECTORY_NAVIGATIONMODE_H
#define _TRAJECTORY_NAVIGATIONMODE_H

#include <string>


class NavigationMode {
	public:
		enum _Value : int {
			Cruise = 0,
			
			IntersectionForward = 110,
			IntersectionLeft = 111,
			IntersectionRight = 112,

			PanicUnsupported = 501,
			PanicException = 502,
			PanicInvalid = 503,
			PanicNoDirection = 550,
		};

		inline constexpr NavigationMode() : m_value(Cruise) {}
		inline constexpr NavigationMode(_Value val) : m_value(val) {}
		
		inline constexpr bool operator==(NavigationMode mode) { return m_value == mode.m_value; }
		inline constexpr bool operator!=(NavigationMode mode) { return m_value != mode.m_value; }
		inline constexpr bool operator==(_Value mode) { return m_value == mode; }
		inline constexpr bool operator!=(_Value mode) { return m_value != mode; }
		inline constexpr operator _Value() const { return m_value; }

		inline constexpr bool is_intersection() const {
			int value = static_cast<int>(m_value);
			return value >= 100 && value < 200;
		}

		inline constexpr bool is_panic() const {
			return static_cast<int>(m_value) >= 500;
		}

		inline constexpr bool is_recoverable_panic() const {
			return static_cast<int>(m_value) >= 550;
		}

		inline std::string what() const {
			switch (m_value) {
				case NavigationMode::Cruise:              return "Cruise";
				case NavigationMode::IntersectionForward: return "Intersection : Forward trajectory";
				case NavigationMode::IntersectionLeft:    return "Intersection : Left turn";
				case NavigationMode::IntersectionRight:   return "Intersection : Right turn";
				case NavigationMode::PanicUnsupported:    return "Panic : Attempted unsupported feature or operation";
				case NavigationMode::PanicException:      return "Panic : Unhandled exception";
				case NavigationMode::PanicInvalid:        return "Panic : Invalid state reported";
				case NavigationMode::PanicNoDirection:    return "Panic : No direction selected";
			}
		}
	
	private:
		_Value m_value;

};


namespace std {
	inline string to_string(NavigationMode mode) {
		switch (mode) {
			case NavigationMode::Cruise:              return "Cruise";
			case NavigationMode::IntersectionForward: return "IntersectionForward";
			case NavigationMode::IntersectionLeft:    return "IntersectionLeft";
			case NavigationMode::IntersectionRight:   return "IntersectionRight";
			case NavigationMode::PanicUnsupported:    return "PanicUnsupported";
			case NavigationMode::PanicException:      return "PanicException";
			case NavigationMode::PanicInvalid:        return "PanicInvalid";
			case NavigationMode::PanicNoDirection:    return "PanicNoDirection";
		}
	}
}

#endif