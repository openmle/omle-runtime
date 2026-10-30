#ifndef OMLE_STATUS_H_
#define OMLE_STATUS_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "omle/port.h"

namespace omle::rt {

// -----------------------------------------------------------------------
// ErrorCode — fine-grained classification of runtime errors.
// -----------------------------------------------------------------------
enum class ErrorCode : uint8_t {
  // Load-time
  FileNotFound = 1,
  FileReadError = 2,
  ParseError = 3,
  InvalidGraph = 4,
  VerificationFailed = 5,

  // Inference-time
  MissingInput = 10,
  OutputNotProduced = 11,
  OutputNotFound = 12,
  UnknownOperator = 13,
  InvalidArgument = 14,
};

// -----------------------------------------------------------------------
// Status — success or typed error (code + message).
//
// Layout: a single owning pointer. Null pointer == success, so a
// default-constructed Status is one null-write on the stack and ok()
// is one null-check. The error payload is heap-allocated only on the
// failure path.
// -----------------------------------------------------------------------
class [[nodiscard]] Status {
  struct Payload {
    ErrorCode code;
    std::string message;
  };

 public:
  Status() noexcept = default;

  Status(ErrorCode code, std::string_view message)
      : err_(std::make_unique<Payload>(Payload{code, std::string(message)})) {}

  Status(const Status& other)
      : err_(other.err_ ? std::make_unique<Payload>(*other.err_) : nullptr) {}

  Status& operator=(const Status& other) {
    if (this != &other) {
      err_ = other.err_ ? std::make_unique<Payload>(*other.err_) : nullptr;
    }
    return *this;
  }

  Status(Status&&) noexcept = default;
  Status& operator=(Status&&) noexcept = default;

  [[nodiscard]] bool ok() const noexcept { return err_ == nullptr; }

  ErrorCode code() const noexcept {
    return err_ ? err_->code : static_cast<ErrorCode>(0);
  }

  const std::string& message() const noexcept {
    static const std::string kEmpty;
    return err_ ? err_->message : kEmpty;
  }

  // Identity accessors so generic propagation macros work uniformly
  // on Status and StatusOr.
  Status status() const& { return *this; }
  Status status() && noexcept { return std::move(*this); }

 private:
  std::unique_ptr<Payload> err_;  // null == success
};

// -----------------------------------------------------------------------
// StatusOr<T> — T on success, Status error on failure.
//
// Backed by std::variant<T, Status>: only one alternative is live at a
// time, so storage is max(sizeof(T), sizeof(Status)) plus a discriminant
// — substantially smaller than separate optional<T>/optional<code>/string
// members held simultaneously.
// -----------------------------------------------------------------------
template <typename T>
class [[nodiscard]] StatusOr {
  using Storage = std::variant<T, Status>;

 public:
  // Success — implicit to allow returning T directly.
  StatusOr(T value) noexcept(std::is_nothrow_move_constructible_v<T>)  // NOLINT
      : data_(std::in_place_index<0>, std::move(value)) {}

  // Error from code + message.
  StatusOr(ErrorCode code, std::string_view message)
      : data_(std::in_place_index<1>, Status(code, message)) {
    assert(!std::get<1>(data_).ok() && "error StatusOr requires nonzero code");
  }

  // Error from Status. Caller is expected to pass a non-ok Status.
  StatusOr(Status s) noexcept  // NOLINT
      : data_(std::in_place_index<1>, std::move(s)) {
    assert(!std::get<1>(data_).ok() &&
           "StatusOr constructed from ok Status — no value provided");
  }

  [[nodiscard]] bool ok() const noexcept { return data_.index() == 0; }

  T& operator*() & noexcept { return *std::get_if<0>(&data_); }
  const T& operator*() const& noexcept { return *std::get_if<0>(&data_); }
  T operator*() && noexcept { return std::move(*std::get_if<0>(&data_)); }
  T* operator->() noexcept { return std::get_if<0>(&data_); }
  const T* operator->() const noexcept { return std::get_if<0>(&data_); }

  // Checked access — throws std::runtime_error if not ok().
  T& value() & {
    if (!ok()) OMLE_UNLIKELY
    throw std::runtime_error(std::get<1>(data_).message());
    return std::get<0>(data_);
  }
  const T& value() const& {
    if (!ok()) OMLE_UNLIKELY
    throw std::runtime_error(std::get<1>(data_).message());
    return std::get<0>(data_);
  }
  T value() && {
    if (!ok()) OMLE_UNLIKELY
    throw std::runtime_error(std::get<1>(data_).message());
    return std::get<0>(std::move(data_));
  }

  ErrorCode code() const noexcept {
    if (auto* s = std::get_if<1>(&data_)) return s->code();
    return static_cast<ErrorCode>(0);
  }

  const std::string& message() const noexcept {
    static const std::string kEmpty;
    if (auto* s = std::get_if<1>(&data_)) return s->message();
    return kEmpty;
  }

  Status status() const& {
    if (ok()) return {};
    return std::get<1>(data_);
  }
  Status status() && noexcept {
    if (ok()) return {};
    return std::get<1>(std::move(data_));
  }

 private:
  Storage data_;
};

}  // namespace omle::rt

#endif  // OMLE_STATUS_H_
